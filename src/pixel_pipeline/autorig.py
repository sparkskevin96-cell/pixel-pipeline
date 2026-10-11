"""Auto-rig: a skeleton, skin weights and animations for any standing humanoid model.

No Mixamo, no upload: everything is worked out from the model itself.

1. The model's front silhouette (an orthographic view, like a T-pose reference sheet) is
   thinned to a one-pixel skeleton. Its ends are the head, hands and feet; where the leg
   paths split is the pelvis, where the arm paths leave the spine is the shoulder line.
   Elbows and knees sit part-way along the limbs; each joint's depth is the middle of the
   body there. Parts the silhouette can't separate (arms against the body, a long robe)
   fall back to standard human proportions.
2. Every vertex follows its nearest bones (inverse distance, up to 4 bones).
3. Animations are made here (idle, walk, run, attack, hurt, death), arms brought down from a
   T/A pose, feet kept on the ground. The result is a normal rigged glTF: turnaround
   --animate renders it in 4/8/16 directions, and Godot or Blender open it as is.

Built for game sprites: at 32-96 px the rough skinning near armpits doesn't show. Works best
on a character standing in an A- or T-pose, facing +z (front), arms clear of the body.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .anim3d import GlbWriter

JOINTS = (("hips", None), ("spine", "hips"), ("chest", "spine"), ("neck", "chest"),
          ("head", "neck"),
          ("upper_arm_l", "chest"), ("forearm_l", "upper_arm_l"), ("hand_l", "forearm_l"),
          ("upper_arm_r", "chest"), ("forearm_r", "upper_arm_r"), ("hand_r", "forearm_r"),
          ("thigh_l", "hips"), ("shin_l", "thigh_l"), ("foot_l", "shin_l"),
          ("thigh_r", "hips"), ("shin_r", "thigh_r"), ("foot_r", "shin_r"))
NAMES = [n for n, _ in JOINTS]
PARENT = dict(JOINTS)
ACTIONS = ("idle", "walk", "run", "attack", "hurt", "death")
LOOPS = {"idle": True, "walk": True, "run": True, "attack": False, "hurt": False, "death": False}
DURATION = {"idle": 1.6, "walk": 1.0, "run": 0.62, "attack": 0.9, "hurt": 0.55, "death": 1.4}
# _l = the character's left = +x (glTF: +y up, front +z).


@dataclass
class Skeleton:
    joints: dict[str, np.ndarray]           # joint name -> position (model space, y up)
    tips: dict[str, np.ndarray]             # head_top, finger_l/r, toe_l/r
    height: float
    notes: list[str] = field(default_factory=list)


@dataclass
class RigResult:
    path: Path
    skeleton: Skeleton
    preview: np.ndarray                    # RGBA: front silhouette with the bones drawn
    clips: list[str]


# --------------------------------------------------------------------------- the model

def _load(model, up: str = "y", yaw: float = 0.0):
    """Meshes of a model, standing on y = 0 around the y axis, front +z.
    Returns [(trimesh mesh, positions)], offset applied to every mesh alike."""
    from .turnaround import _trimesh
    trimesh = _trimesh()
    loaded = trimesh.load(str(model), process=False)
    meshes = [loaded] if isinstance(loaded, trimesh.Trimesh) else loaded.dump()
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if not meshes:
        raise ValueError(f"No triangles found in {model}")
    out = []
    for m in meshes:
        v = np.asarray(m.vertices, np.float64)
        if up == "z":
            v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
        if yaw:
            t = math.radians(yaw)
            v = np.stack([v[:, 0] * math.cos(t) + v[:, 2] * math.sin(t), v[:, 1],
                          -v[:, 0] * math.sin(t) + v[:, 2] * math.cos(t)], 1)
        out.append((m, v))
    allv = np.concatenate([v for _, v in out])
    lo, hi = allv.min(0), allv.max(0)
    shift = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return [(m, v - shift) for m, v in out]


def _surface_points(parts, count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tris = np.concatenate([v[np.asarray(m.faces)] for m, v in parts])
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    face = rng.choice(len(tris), size=count, p=area / area.sum())
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    return (tris[face] * bary[..., None]).sum(1)


# --------------------------------------------------------------------------- silhouette

class _Grid:
    """Front silhouette: mask[row, col], depth range per cell, cell <-> model x/y."""

    def __init__(self, pts: np.ndarray, rows: int = 220, pad: int = 3):
        self.height = float(pts[:, 1].max())
        self.cell = self.height / rows
        self.pad, self.res = pad, rows
        self.xmin = float(pts[:, 0].min())
        cols = int((pts[:, 0].max() - self.xmin) / self.cell) + 1
        shape = (rows + 2 * pad, cols + 2 * pad)
        r = pad + rows - 1 - np.clip((pts[:, 1] / self.cell).astype(int), 0, rows - 1)
        c = pad + np.clip(((pts[:, 0] - self.xmin) / self.cell).astype(int), 0, cols - 1)
        lin = r * shape[1] + c
        zmin = np.full(shape[0] * shape[1], np.inf)
        zmax = np.full(shape[0] * shape[1], -np.inf)
        np.minimum.at(zmin, lin, pts[:, 2])
        np.maximum.at(zmax, lin, pts[:, 2])
        mask = np.isfinite(zmin).reshape(shape)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        self.mask = _join_pieces(_fill_small_holes(mask, 0.004))
        self.zmin, self.zmax = zmin.reshape(shape), zmax.reshape(shape)

    def xy(self, r, c) -> tuple[float, float]:
        return (self.xmin + (c - self.pad + 0.5) * self.cell,
                (self.res - 1 - (r - self.pad) + 0.5) * self.cell)

    def rc(self, x, y) -> tuple[int, int]:
        r = self.pad + self.res - 1 - int(np.clip(y / self.cell, 0, self.res - 1))
        c = self.pad + int(np.clip((x - self.xmin) / self.cell, 0, self.mask.shape[1] - 2 * self.pad - 1))
        return r, c

    def depth(self, x, y) -> float:
        """Middle of the body's depth at (x, y) in the front view (nearest filled cell)."""
        r, c = self.rc(x, y)
        for rad in range(0, 8):
            r0, r1 = max(0, r - rad), r + rad + 1
            c0, c1 = max(0, c - rad), c + rad + 1
            lo, hi = self.zmin[r0:r1, c0:c1], self.zmax[r0:r1, c0:c1]
            ok = np.isfinite(lo)
            if ok.any():
                return float((lo[ok].min() + hi[ok].max()) / 2)
        return 0.0

    def run_width(self, y, x_center) -> float:
        """Width of the silhouette run at height y that contains x_center (model units)."""
        r, c = self.rc(x_center, y)
        row = self.mask[r]
        if not row[c]:
            near = np.nonzero(row)[0]
            if not len(near):
                return 0.0
            c = int(near[np.argmin(np.abs(near - c))])
        left = c
        while left > 0 and row[left - 1]:
            left -= 1
        right = c
        while right < len(row) - 1 and row[right + 1]:
            right += 1
        return (right - left + 1) * self.cell


def _fill_small_holes(mask: np.ndarray, frac: float) -> np.ndarray:
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    limit = frac * mask.sum()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area < limit:
            out[labels == i] = True
    return out


def _join_pieces(mask: np.ndarray) -> np.ndarray:
    """Bridge separate silhouette pieces (a floating arm or accessory) to the biggest one."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    out = mask.astype(np.uint8)
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    my, mx = np.nonzero(labels == main)
    for i in range(1, n):
        if i == main or stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        py, px = np.nonzero(labels == i)
        k = np.linspace(0, len(py) - 1, min(len(py), 200)).astype(int)
        d = (py[k, None] - my[None, ::7]) ** 2 + (px[k, None] - mx[None, ::7]) ** 2
        a, b = np.unravel_index(int(np.argmin(d)), d.shape)
        cv2.line(out, (int(px[k[a]]), int(py[k[a]])), (int(mx[b * 7]), int(my[b * 7])), 1, 2)
    return out.astype(bool)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: a one-pixel-wide skeleton of a binary mask."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            core = img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(x.astype(np.int16) for x in ring[:8])
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
            else:
                c1, c2 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
            rm = (core == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if rm.any():
                core[rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


class _Graph:
    """8-connected skeleton pixels as a graph. Ends and junctions use the crossing number
    (0 -> 1 changes around the 8 neighbours), so pixel staircases on slanted lines don't
    count as branches."""

    def __init__(self, skel: np.ndarray):
        self.pix = {(int(r), int(c)) for r, c in zip(*np.nonzero(skel))}
        self.adj = {p: [q for q in ((p[0] + dr, p[1] + dc) for dr, dc in RING) if q in self.pix]
                    for p in self.pix}

    def crossings(self, p) -> int:
        on = [((p[0] + dr, p[1] + dc) in self.pix) for dr, dc in RING]
        return sum(1 for i in range(8) if not on[i] and on[(i + 1) % 8])

    def ends(self) -> list:
        return [p for p in self.pix if self.crossings(p) == 1 and len(self.adj[p]) <= 2]

    def path(self, a, b) -> list:
        prev = {a: None}
        todo = deque([a])
        while todo:
            p = todo.popleft()
            if p == b:
                break
            for q in self.adj[p]:
                if q not in prev:
                    prev[q] = p
                    todo.append(q)
        if b not in prev:
            return []
        out, p = [], b
        while p is not None:
            out.append(p)
            p = prev[p]
        return out[::-1]

    def _remove(self, pts):
        for q in pts:
            self.pix.discard(q)
            for n in self.adj.pop(q, []):
                if n in self.adj and q in self.adj[n]:
                    self.adj[n].remove(q)

    def prune(self, min_len: int):
        """Drop short side branches (silhouette noise), twice over."""
        for _ in range(2):
            for e in self.ends():
                if e not in self.pix:
                    continue
                branch, seen, p = [e], {e}, e
                while len(branch) <= min_len:
                    nxt = [q for q in self.adj.get(p, []) if q not in seen]
                    if not nxt:
                        break
                    p = min(nxt, key=lambda q: abs(q[0] - p[0]) + abs(q[1] - p[1]))  # 4-neighbours first
                    if self.crossings(p) >= 3:
                        self._remove(branch)
                        break
                    seen.update(self.adj[branch[-1]])
                    branch.append(p)


def _along(grid, path: list, frac: float) -> np.ndarray:
    """The point a fraction of the way along a pixel path (by length), as model (x, y)."""
    pts = np.array([grid.xy(r, c) for r, c in path])
    if len(pts) == 1:
        return pts[0]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = frac * cum[-1]
    i = int(np.clip(np.searchsorted(cum, t) - 1, 0, len(seg) - 1))
    u = (t - cum[i]) / max(seg[i], 1e-9)
    return pts[i] + u * (pts[i + 1] - pts[i])


def _at_y(grid, path: list, y: float) -> np.ndarray:
    pts = np.array([grid.xy(r, c) for r, c in path])
    return pts[int(np.argmin(np.abs(pts[:, 1] - y)))]


# --------------------------------------------------------------------------- skeleton

def find_skeleton(pts: np.ndarray) -> tuple[Skeleton, _Grid, np.ndarray]:
    """Joint positions of a standing humanoid from its surface points (normalised)."""
    grid = _Grid(pts)
    H = grid.height
    notes: list[str] = []
    skel = thin(grid.mask)
    g = _Graph(skel)
    g.prune(max(3, int(0.05 * grid.res)))
    mid_rows = (pts[:, 1] > 0.3 * H) & (pts[:, 1] < 0.6 * H)
    xc = float(np.median(pts[mid_rows, 0])) if mid_rows.any() else 0.0
    ends = [(p, *grid.xy(*p)) for p in g.ends()]

    # head: highest end near the middle
    central = [e for e in ends if abs(e[1] - xc) < 0.2 * H and e[2] > 0.6 * H]
    head = max(central, key=lambda e: e[2]) if central else None
    if head is None:
        rows = np.nonzero(grid.mask.any(1))[0]
        top_r = rows.min()
        cols = np.nonzero(grid.mask[top_r])[0]
        head_px = min(g.pix, key=lambda p: (p[0], abs(p[1] - cols.mean()))) if g.pix else (top_r, int(cols.mean()))
        head = (head_px, *grid.xy(*head_px))
        notes.append("head: top of the silhouette")
    head_top = np.array([head[1], H * 0.995])

    # feet: lowest ends left and right of the middle
    low = [e for e in ends if e[2] < 0.3 * H]
    foot = {"l": min((e for e in low if e[1] > xc), key=lambda e: e[2], default=None),
            "r": min((e for e in low if e[1] < xc), key=lambda e: e[2], default=None)}
    leg_paths = {s: g.path(head[0], foot[s][0]) if foot[s] else [] for s in "lr"}
    joints: dict[str, np.ndarray] = {}
    tips: dict[str, np.ndarray] = {"head_top": head_top}

    if all(leg_paths.values()):
        a, b = leg_paths["l"], leg_paths["r"]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        fork_y = grid.xy(*a[max(0, k - 1)])[1]
        spine_path = a[:k]
        # the crotch: lowest filled cell between the legs (the skeleton forks a little higher)
        _, c = grid.rc(xc, 0)
        filled = [r for r in range(grid.mask.shape[0]) if grid.mask[r, c] and grid.xy(r, c)[1] > 0.15 * H]
        crotch_y = min((grid.xy(r, c)[1] for r in filled), default=fork_y)
        crotch_y = min(crotch_y, fork_y)
        pelvis_y = min(crotch_y + 0.03 * H, 0.65 * H)
        legs = {s: leg_paths[s][k - 1:] for s in "lr"}
    else:
        pelvis_y = 0.5 * H
        spine_path = g.path(head[0], min(g.pix, key=lambda p: abs(grid.xy(*p)[1] - pelvis_y)
                                         + abs(grid.xy(*p)[0] - xc))) if g.pix else []
        legs = {}
        notes.append("legs: not separate (robe?), standard proportions")
    pelvis = np.array([xc, pelvis_y])

    # arms: the most sideways ends between the knees and the head
    arm_end = {}
    for s, sign in (("l", 1), ("r", -1)):
        side = [e for e in ends if (e[1] - xc) * sign > 0.08 * H and 0.25 * H < e[2] < 0.97 * H
                and (not foot[s] or e[0] != foot[s][0])]
        arm_end[s] = max(side, key=lambda e: (e[1] - xc) * sign) if side else None
    waist = min((grid.run_width(y, xc) for y in np.linspace(pelvis_y + 0.05 * H, pelvis_y + 0.2 * H, 6)),
                default=0.25 * H) / 2
    chest_half = max(0.06 * H, min(waist * 1.15, 0.2 * H))
    shoulder_y_guess = pelvis_y + 0.55 * (head_top[1] - pelvis_y) * 0.62 / 0.55 * 0.55
    arm_paths, junction_y = {}, []
    for s, sign in (("l", 1), ("r", -1)):
        if arm_end[s] is None:
            continue
        p = g.path(head[0], arm_end[s][0])
        k = 0
        sp = set(spine_path)
        while k < len(p) and p[k] in sp:
            k += 1
        arm = p[max(0, k - 1):]
        if len(arm) < 4:
            continue
        arm_paths[s] = arm
        junction_y.append(grid.xy(*arm[0])[1])
    shoulder_y = float(np.mean(junction_y)) if junction_y else shoulder_y_guess
    shoulder_y = float(np.clip(shoulder_y, pelvis_y + 0.15 * H, head_top[1] - 0.1 * H))

    # spine column
    def spine_x(y):
        return float(_at_y(grid, spine_path, y)[0]) if spine_path else xc
    # neck: narrowest part between the shoulders and the head top
    ys = np.linspace(shoulder_y, shoulder_y + 0.7 * (head_top[1] - shoulder_y), 12)
    widths = [grid.run_width(y, spine_x(y)) for y in ys]
    neck_top = float(ys[int(np.argmin(widths))])
    if neck_top <= shoulder_y + 0.02 * H:
        neck_top = shoulder_y + 0.3 * (head_top[1] - shoulder_y)
    span = shoulder_y - pelvis_y
    for name, y in (("hips", pelvis_y), ("spine", pelvis_y + 0.35 * span),
                    ("chest", pelvis_y + 0.75 * span), ("neck", shoulder_y + 0.4 * (neck_top - shoulder_y)),
                    ("head", neck_top)):
        joints[name] = np.array([spine_x(y), y])
    joints["hips"][0] = xc

    for s, sign in (("l", 1), ("r", -1)):
        arm = arm_paths.get(s)
        if arm:
            pts2 = np.array([grid.xy(r, c) for r, c in arm])
            out = np.nonzero((pts2[:, 0] - xc) * sign >= chest_half)[0]
            start = int(out[0]) if len(out) else 0
            limb = arm[start:] if len(arm) - start >= 3 else arm
            shoulder = np.array(grid.xy(*limb[0]))
            shoulder[1] = min(shoulder[1], shoulder_y)
            finger = np.array(grid.xy(*limb[-1]))
            joints[f"upper_arm_{s}"] = shoulder
            joints[f"forearm_{s}"] = _along(grid, limb, 0.5)
            joints[f"hand_{s}"] = _along(grid, limb, 0.86)
            tips[f"finger_{s}"] = finger
        else:
            notes.append(f"arm {s}: not separate, hanging at the side")
            shoulder = np.array([xc + sign * chest_half, shoulder_y])
            joints[f"upper_arm_{s}"] = shoulder
            joints[f"forearm_{s}"] = shoulder + [sign * 0.02 * H, -0.17 * H]
            joints[f"hand_{s}"] = shoulder + [sign * 0.035 * H, -0.32 * H]
            tips[f"finger_{s}"] = shoulder + [sign * 0.04 * H, -0.4 * H]

        leg = legs.get(s)
        if leg:
            hip_pt = _at_y(grid, leg, pelvis_y - 0.06 * H)
            hip = np.array([hip_pt[0], pelvis_y - 0.02 * H])
            ankle_y = float(np.clip(grid.xy(*leg[-1])[1], 0.035 * H, 0.06 * H))
            joints[f"thigh_{s}"] = hip
            joints[f"shin_{s}"] = _at_y(grid, leg, (hip[1] + ankle_y) / 2)
            joints[f"foot_{s}"] = _at_y(grid, leg, ankle_y)
        else:
            hx = xc + sign * max(0.05 * H, waist * 0.5)
            joints[f"thigh_{s}"] = np.array([hx, pelvis_y - 0.02 * H])
            joints[f"shin_{s}"] = np.array([hx, (pelvis_y + 0.05 * H) / 2])
            joints[f"foot_{s}"] = np.array([hx, 0.05 * H])
        joints[f"foot_{s}"][1] = max(joints[f"foot_{s}"][1], 0.03 * H)
        tips[f"toe_{s}"] = np.array([joints[f"foot_{s}"][0], 0.01 * H])

    def three(xy, z=None):
        return np.array([xy[0], xy[1], grid.depth(*xy) if z is None else z])
    j3 = {n: three(joints[n]) for n in NAMES}
    for s in "lr":   # toes point forward from the ankle
        j3[f"foot_{s}"][2] = grid.depth(*joints[f"shin_{s}"])
    t3 = {n: three(p) for n, p in tips.items()}
    for s in "lr":
        t3[f"toe_{s}"][2] = j3[f"foot_{s}"][2] + 0.07 * H
    return Skeleton(j3, t3, H, notes), grid, skel


# --------------------------------------------------------------------------- skinning

def _bones(sk: Skeleton) -> list[tuple[str, np.ndarray, np.ndarray]]:
    j, t = sk.joints, sk.tips
    segs = [("hips", j["hips"], j["spine"]), ("hips", j["hips"], j["thigh_l"]),
            ("hips", j["hips"], j["thigh_r"]), ("spine", j["spine"], j["chest"]),
            ("chest", j["chest"], j["neck"]), ("chest", j["chest"], j["upper_arm_l"]),
            ("chest", j["chest"], j["upper_arm_r"]), ("neck", j["neck"], j["head"]),
            ("head", j["head"], t["head_top"])]
    for s in "lr":
        segs += [(f"upper_arm_{s}", j[f"upper_arm_{s}"], j[f"forearm_{s}"]),
                 (f"forearm_{s}", j[f"forearm_{s}"], j[f"hand_{s}"]),
                 (f"hand_{s}", j[f"hand_{s}"], t[f"finger_{s}"]),
                 (f"thigh_{s}", j[f"thigh_{s}"], j[f"shin_{s}"]),
                 (f"shin_{s}", j[f"shin_{s}"], j[f"foot_{s}"]),
                 (f"foot_{s}", j[f"foot_{s}"], t[f"toe_{s}"])]
    return segs


def skin_weights(vertices: np.ndarray, sk: Skeleton, influences: int = 4,
                 power: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """(joint indices, weights) per vertex: nearest bones by distance to the bone segment."""
    dist = np.full((len(vertices), len(NAMES)), np.inf)
    for owner, a, b in _bones(sk):
        ab = b - a
        t = np.clip(((vertices - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        d = np.linalg.norm(vertices - (a + t[:, None] * ab), axis=1)
        k = NAMES.index(owner)
        dist[:, k] = np.minimum(dist[:, k], d)
    w = 1.0 / (dist / sk.height + 0.01) ** power
    idx = np.argsort(-w, axis=1)[:, :influences]
    top = np.take_along_axis(w, idx, 1)
    top /= top.sum(1, keepdims=True)
    return idx.astype(np.uint8), top.astype(np.float32)


# --------------------------------------------------------------------------- animations

def _q(axis: str, deg: float) -> np.ndarray:
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([v[0] * math.sin(h), v[1] * math.sin(h), v[2] * math.sin(h), math.cos(h)])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _slerp(a, b, u):
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
    return q / np.linalg.norm(q)


def _smooth(u):
    return u * u * (3 - 2 * u)


class _Moves:
    """Poses of the procedural animations on one skeleton.
    A pose = {joint: [(axis, degrees), ...]} applied right to left, plus a hips offset."""

    def __init__(self, sk: Skeleton, weapon: str = "r"):
        self.sk, self.H, self.w = sk, sk.height, weapon
        j = sk.joints
        self.lower = {}
        for s, sign in (("l", 1), ("r", -1)):   # bring the arms down to ~12 degrees
            d = j[f"hand_{s}"] - j[f"upper_arm_{s}"]
            alpha = math.degrees(math.atan2(abs(d[0]), -d[1]))
            self.lower[s] = -sign * max(0.0, alpha - 12.0)
        self.leg = float(np.mean([j[f"thigh_{s}"][1] - j[f"foot_{s}"][1] + 0.04 * self.H for s in "lr"]))
        back = j["hips"][2] - min(sk.tips["head_top"][2], j["chest"][2] - 0.08 * self.H)
        self.back = float(max(0.05 * self.H, back))

    def arm(self, s, swing=0.0, out=0.0, bend=-10.0):
        sign = 1 if s == "l" else -1
        return {f"upper_arm_{s}": [("x", swing), ("z", self.lower[s] + sign * out)],
                f"forearm_{s}": [("x", bend)]}

    def drop(self, angle):
        """Hips drop that keeps a straight stance leg on the ground at this swing angle."""
        return -self.leg * (1 - math.cos(math.radians(abs(angle))))

    def pose(self, action: str, p: float):
        s, c = math.sin(2 * math.pi * p), math.cos(2 * math.pi * p)
        if action == "idle":
            b = (1 - c) / 2
            return ({"spine": [("x", 1.5 * b)], "chest": [("x", 1.0 * b)], "head": [("x", -1.5 * b)],
                     **self.arm("l", 2 * b, 1 * b, -8), **self.arm("r", 2 * b, 1 * b, -8)},
                    np.array([0, -0.006 * self.H * b, 0]))
        if action in ("walk", "run"):
            run = action == "run"
            amp, knee, arm, bend = (40, 75, 40, -75) if run else (24, 45, 18, -15)
            th = amp * s
            rot = {"thigh_l": [("x", -th)], "thigh_r": [("x", th)],
                   "shin_l": [("x", 5 + knee * max(0.0, c) ** 2)],
                   "shin_r": [("x", 5 + knee * max(0.0, -c) ** 2)],
                   "hips": [("y", 4 * s)], "spine": [("y", -4 * s), ("x", 12 if run else 2)],
                   **self.arm("l", arm * s, 0, bend), **self.arm("r", -arm * s, 0, bend)}
            lift = 0.012 * self.H * abs(c) if run else 0.0
            return rot, np.array([0, self.drop(th) + lift, 0])
        return self._keyed(action, p)

    def _keyed(self, action, p):
        w, o = self.w, "l" if self.w == "r" else "r"
        sw = -1 if w == "r" else 1          # spine twist that pulls the weapon shoulder back
        keys = {
            "attack": [(0.0, {}, 0), (0.35, {"spine": [("y", 25 * sw), ("x", -8)],
                                             **self.arm(w, -190, 0, -70), **self.arm(o, 20, 10, -20),
                                             "thigh_" + w: [("x", 10)], "shin_" + w: [("x", 10)]}, 0),
                       (0.55, {"spine": [("y", -25 * sw), ("x", 15)], **self.arm(w, -55, 0, -10),
                               **self.arm(o, -15, 10, -30), "thigh_" + o: [("x", -25)],
                               "shin_" + o: [("x", 20)], "thigh_" + w: [("x", 15)]}, -0.02),
                       (1.0, {}, 0)],
            "hurt": [(0.0, {}, 0), (0.25, {"spine": [("x", -15)], "chest": [("x", -10)],
                                           "head": [("x", -15)], **self.arm("l", 15, 25, -30),
                                           **self.arm("r", 15, 25, -30), "thigh_l": [("x", -8)],
                                           "thigh_r": [("x", -8)], "shin_l": [("x", 10)],
                                           "shin_r": [("x", 10)]}, -0.02),
                     (1.0, {}, 0)],
            "death": [(0.0, {}, 0),
                      (0.25, {"thigh_l": [("x", -35)], "thigh_r": [("x", -35)], "shin_l": [("x", 60)],
                              "shin_r": [("x", 60)], "spine": [("x", 20)], "head": [("x", 15)],
                              **self.arm("l", 0, 15, -20), **self.arm("r", 0, 15, -20)}, -0.09),
                      (0.6, {"hips": [("x", -45)], "thigh_l": [("x", -20)], "thigh_r": [("x", -20)],
                             "shin_l": [("x", 30)], "shin_r": [("x", 30)], **self.arm("l", 0, 40, -10),
                             **self.arm("r", 0, 40, -10)}, "mid"),
                      (1.0, {"hips": [("x", -88)], **self.arm("l", 0, 60, 0), **self.arm("r", 0, 60, 0),
                             "head": [("x", -5)]}, "flat")],
        }[action]
        rest_y = self.sk.joints["hips"][1]
        flat = self.back + 0.01 * self.H - rest_y
        for i in range(len(keys) - 1):
            if keys[i][0] <= p <= keys[i + 1][0] or i == len(keys) - 2:
                (t0, a, ya), (t1, b, yb) = keys[i], keys[i + 1]
                break
        u = _smooth(min(1.0, max(0.0, (p - t0) / max(t1 - t0, 1e-9))))

        def y(v):
            if v == "flat":
                return flat
            if v == "mid":
                return 0.55 * flat
            return v * self.H
        names = set(a) | set(b)
        rot = {n: ("blend", self._q(a.get(n, self._rest(n))), self._q(b.get(n, self._rest(n))), u)
               for n in names}
        return rot, np.array([0, y(ya) + (y(yb) - y(ya)) * u, 0])

    def _rest(self, n):
        for s in "lr":
            if n == f"upper_arm_{s}":
                return [("z", self.lower[s])]
            if n == f"forearm_{s}":
                return [("x", -10)]
        return []

    @staticmethod
    def _q(spec):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        for axis, deg in spec:
            q = _qmul(q, _q(axis, deg))
        return q

    def quats(self, action, p) -> tuple[dict[str, np.ndarray], np.ndarray]:
        rot, off = self.pose(action, p)
        out = {}
        for n in NAMES:
            spec = rot.get(n)
            if spec is None:
                spec = self._rest(n)
            if isinstance(spec, tuple) and spec and spec[0] == "blend":
                out[n] = _slerp(spec[1], spec[2], spec[3])
            else:
                out[n] = self._q(spec)
        return out, off


# --------------------------------------------------------------------------- glTF out

def _srgb_to_linear(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def write_rigged(parts, sk: Skeleton, out_glb, actions=ACTIONS, weapon: str = "r",
                 keys: int = 25) -> Path:
    """Mesh + skeleton + skin + animations as one .glb."""
    moves = _Moves(sk, weapon)
    clips = []
    for action in actions:
        dur = DURATION[action]
        times = np.linspace(0, dur, keys)
        clips.append((action, times, [moves.quats(action, t / dur) for t in times], LOOPS[action]))
    return write_skinned(parts, JOINTS, sk.joints, lambda v: skin_weights(v, sk), clips, out_glb)


def write_skinned(parts, joints, positions: dict, weights, clips, out_glb) -> Path:
    """Meshes + a world-aligned skeleton + skin + rotation animations as one .glb.
    joints: [(name, parent)], parents first, the first one is the root. positions: name -> rest
    position. weights(vertices) -> (joint indices, weights). clips: [(name, times, poses, loop)]
    with poses = [({joint: quaternion}, root offset)]. Vertex colours and base colour textures
    are kept."""
    from .turnaround import _texture
    g = GlbWriter()
    j = positions
    names = [n for n, _ in joints]
    parent_of = dict(joints)
    root = names[0]
    nodes = [{"name": "Armature", "children": [1]}]
    index = {n: i + 1 for i, n in enumerate(names)}
    for n in names:
        parent = parent_of[n]
        offset = j[n] - (j[parent] if parent else 0)
        node = {"name": n, "translation": [float(x) for x in offset]}
        kids = [index[c] for c, p in joints if p == n]
        if kids:
            node["children"] = kids
        nodes.append(node)
    ibm = np.tile(np.eye(4, dtype=np.float32), (len(names), 1, 1))
    for i, n in enumerate(names):
        ibm[i, :3, 3] = -j[n]
    g.doc["skins"] = [{"joints": [index[n] for n in names], "skeleton": index[root],
                       "inverseBindMatrices": g.accessor(ibm.transpose(0, 2, 1).reshape(-1, 16), "MAT4")}]
    prims, materials, images, textures = [], [], [], []
    for m, v in parts:
        idx, wts = weights(v)
        attrs = {"POSITION": g.accessor(v.astype(np.float32), "VEC3"),
                 "JOINTS_0": g.accessor(idx, "VEC4", 5121),
                 "WEIGHTS_0": g.accessor(wts, "VEC4")}
        kind = getattr(m.visual, "kind", None)
        mat: dict = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if kind == "texture" and getattr(m.visual, "uv", None) is not None:
            tex, factor = _texture(m)
            uv = np.asarray(m.visual.uv, np.float64).copy()
            uv[:, 1] = 1 - uv[:, 1]                       # trimesh: bottom-left; glTF: top-left
            attrs["TEXCOORD_0"] = g.accessor(uv.astype(np.float32), "VEC2")
            mat["pbrMetallicRoughness"]["baseColorFactor"] = [float(x) for x in factor]
            if tex is not None:
                import io
                buf = io.BytesIO()
                Image.fromarray(tex).save(buf, "PNG")
                images.append({"bufferView": g.view(buf.getvalue()), "mimeType": "image/png"})
                textures.append({"source": len(images) - 1})
                mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
        elif kind in ("vertex", "face"):
            vc = np.asarray(m.visual.vertex_colors, np.float64)[:, :3]
            lin = np.concatenate([_srgb_to_linear(vc), np.ones((len(vc), 1))], 1)
            attrs["COLOR_0"] = g.accessor(lin.astype(np.float32), "VEC4")
        materials.append(mat)
        faces = np.asarray(m.faces, np.uint32).reshape(-1)
        prims.append({"attributes": attrs, "indices": g.accessor(faces, "SCALAR", 5125),
                      "material": len(materials) - 1})
    g.doc["materials"] = materials
    if images:
        g.doc["images"], g.doc["textures"] = images, textures
    g.doc["meshes"] = [{"name": "body", "primitives": prims}]
    nodes.append({"name": "body", "mesh": 0, "skin": 0})
    g.doc["nodes"] = nodes
    g.doc["scenes"] = [{"nodes": [0, len(nodes) - 1]}]
    g.doc["scene"] = 0

    anims = []
    for name, times, poses, loop in clips:
        t_acc = g.accessor(np.asarray(times, np.float32), "SCALAR")
        samplers, channels = [], []
        for n in names:
            q = np.array([p[0][n] for p in poses])
            for k in range(1, len(q)):                     # keep neighbours on one hemisphere
                if q[k] @ q[k - 1] < 0:
                    q[k] = -q[k]
            samplers.append({"input": t_acc, "output": g.accessor(q.astype(np.float32), "VEC4")})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": index[n], "path": "rotation"}})
        moved = np.array([j[root] + p[1] for p in poses])
        samplers.append({"input": t_acc, "output": g.accessor(moved.astype(np.float32), "VEC3")})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": index[root], "path": "translation"}})
        anims.append({"name": name, "samplers": samplers, "channels": channels,
                      "extras": {"loop": bool(loop)}})
    g.doc["animations"] = anims
    return g.save(out_glb)


# --------------------------------------------------------------------------- preview + one call

def rig_preview(grid: _Grid, skel_px: np.ndarray, sk: Skeleton, scale: int = 2) -> np.ndarray:
    """Front silhouette (grey), thinned skeleton (dark) and the joints and bones found."""
    m = grid.mask
    img = np.zeros(m.shape + (4,), np.uint8)
    img[m] = (150, 156, 170, 255)
    img[skel_px] = (90, 96, 112, 255)
    im = Image.fromarray(img).resize((m.shape[1] * scale, m.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def px(p):
        r, c = grid.rc(p[0], p[1])
        return ((c + 0.5) * scale, (r + 0.5) * scale)
    colours = {"l": (60, 200, 90), "r": (230, 80, 70)}
    for owner, a, b in _bones(sk):
        side = owner[-1] if owner[-2:] in ("_l", "_r") else None
        d.line([px(a), px(b)], fill=colours.get(side, (250, 210, 60)), width=max(1, scale))
    for n, p in sk.joints.items():
        x, y = px(p)
        d.ellipse([x - 2 * scale, y - 2 * scale, x + 2 * scale, y + 2 * scale], fill=(255, 255, 255),
                  outline=(20, 20, 30))
    return np.asarray(im)


def auto_rig(model, out_glb=None, *, actions=ACTIONS, weapon_hand: str = "right", up: str = "y",
             yaw: float = 0.0, points: int = 400_000) -> RigResult:
    """A standing humanoid model -> rigged, animated .glb (default <name>.rigged.glb)."""
    unknown = [a for a in actions if a not in ACTIONS]
    if unknown:
        raise ValueError(f"Unknown animation(s) {', '.join(unknown)} (have: {', '.join(ACTIONS)})")
    model = Path(model)
    out = Path(out_glb) if out_glb else model.with_name(model.stem + ".rigged.glb")
    parts = _load(model, up, yaw)
    pts = _surface_points(parts, points)
    sk, grid, skel_px = find_skeleton(pts)
    write_rigged(parts, sk, out, actions, "r" if weapon_hand.startswith("r") else "l")
    return RigResult(out, sk, rig_preview(grid, skel_px, sk), list(actions))

"""Auto-rig for four-legged animals (dogs first): the quadruped counterpart of autorig.py.

1. The long axis of the model (x or z) is the body; the head end is the end with more
   shape above the back line (a raised tail is thin). The model is turned so the head faces
   +z, the same "front +z, left +x" as the humanoid rig, so turnaround's S view is the
   animal coming towards the camera and E is it walking right.
2. In the SIDE silhouette (z, y) the legs are what is below the belly: the lowest row where
   one silhouette run spans most of the body is the belly line, leg points below it split
   into a front and a back pair at the empty gap between them, and each pair into left and
   right by depth (near and far legs overlap in a side view). Spine joints sit on the
   body's middle line above the legs, the head is the mass above the back at the front, a
   tail is a run of thin silhouette columns at the back. Failures fall back to standard dog
   proportions (notes say which).
3. Skin = inverse distance^4 to the bone segments, top 4, but a leg only binds vertices on
   its own side (depth) and pair (front/back) and below the body, so legs don't drag the
   opposite leg or the back along.
4. Animations: legs are posed with 2-bone IK in the side plane towards foot targets on the
   ground, so stance feet stay planted: idle, walk (4-beat lateral walk), run (transverse
   gallop), sit, happy (fast tail wag + head bob), lie_down, death (falls on its side).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .autorig import _Grid, _load, _q, _qmul, _smooth, _surface_points, write_skinned

LEGS = ("front_l", "front_r", "back_l", "back_r")
PARTS = ("upper", "lower", "paw")
JOINTS = (("hips", None), ("spine", "hips"), ("chest", "spine"), ("neck", "chest"),
          ("head", "neck"), ("tail1", "hips"), ("tail2", "tail1"),
          *[(f"{leg}_{p}", (("chest" if leg.startswith("front") else "hips") if i == 0
                           else f"{leg}_{PARTS[i - 1]}"))
            for leg in LEGS for i, p in enumerate(PARTS)])
NAMES = [n for n, _ in JOINTS]
PARENT = dict(JOINTS)
ACTIONS = ("idle", "walk", "run", "sit", "happy", "lie_down", "death")
LOOPS = {"idle": True, "walk": True, "run": True, "sit": False, "happy": True,
         "lie_down": False, "death": False}
DURATION = {"idle": 1.6, "walk": 1.0, "run": 0.56, "sit": 1.0, "happy": 0.8,
            "lie_down": 1.2, "death": 1.2}
HEADS = {"+z": 0.0, "-z": 180.0, "+x": -90.0, "-x": 90.0}   # yaw that turns that end to +z
# _l = the animal's left = +x (head +z, up +y), like the humanoid rig.


@dataclass
class QuadSkeleton:
    joints: dict[str, np.ndarray]          # joint name -> position (head +z, left +x, y up)
    tips: dict[str, np.ndarray]            # head_tip, head_top, tail_tip, rump, <leg>_toe
    height: float
    notes: list[str] = field(default_factory=list)
    yaw: float = 0.0                       # degrees the model was turned so the head faces +z
    head: str = "+z"                       # which end of the input was the head
    length: float = 1.0                    # body length along z
    belly: float = 0.5                     # height of the belly line
    pair_z: float = 0.0                    # between the front and back legs
    split_x: dict[str, float] = field(default_factory=dict)   # front/back: left-right split
    has_tail: bool = False
    body: np.ndarray | None = field(default=None, repr=False)  # body surface sample (posing)


@dataclass
class QuadRigResult:
    path: Path
    skeleton: QuadSkeleton
    preview: np.ndarray                    # RGBA: side silhouette with the bones drawn
    clips: list[str]

    @property
    def notes(self) -> list[str]:
        return self.skeleton.notes


def _turn(v: np.ndarray, yaw: float) -> np.ndarray:
    """Same turn as autorig._load(yaw=...): degrees about +y."""
    if not yaw:
        return v
    t = math.radians(yaw)
    return np.stack([v[:, 0] * math.cos(t) + v[:, 2] * math.sin(t), v[:, 1],
                     -v[:, 0] * math.sin(t) + v[:, 2] * math.cos(t)], 1)


# --------------------------------------------------------------------------- orientation

def body_axis_yaw(pts: np.ndarray, min_turn: float = 12.0) -> float:
    """Yaw (degrees, as _turn) that lines a body lying diagonally in x-z up with the nearest of
    the x / z axes; 0 when it already is (within min_turn) or the body isn't clearly long.
    Prompt-to-model animals come from a three-quarter picture, and the model is turned so that
    picture is its front: the wolf's body ran ~45 degrees between x and z, so the long-axis
    test below (x or z) cut it across and the rig + every sprite came out skewed."""
    y = pts[:, 1]
    H = float(y.max() - y.min()) or 1.0
    body = pts[(y > y.min() + 0.3 * H) & (y < y.min() + 0.9 * H)]
    if len(body) < 50:
        return 0.0
    xz = body[:, [0, 2]] - body[:, [0, 2]].mean(0)
    w, v = np.linalg.eigh(np.cov(xz.T))
    if w[1] < 1.3 * max(w[0], 1e-12):                     # round footprint: no clear long axis
        return 0.0
    phi = math.degrees(math.atan2(v[1, 1], v[0, 1]))      # angle from +x towards +z
    snap = phi - 90.0 * round(phi / 90.0)
    return float(snap) if abs(snap) >= min_turn else 0.0


def facing_yaw(pts: np.ndarray, head: str = "auto") -> tuple[float, str, list[str]]:
    """(yaw that turns the head to +z, which end the head was, notes)."""
    if head != "auto":
        if head not in HEADS:
            raise ValueError(f"head must be auto or one of {', '.join(HEADS)}")
        return HEADS[head], head, []
    ax = 0 if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 2]) else 2
    a, y = pts[:, ax], pts[:, 1]
    lo, hi = float(a.min()), float(a.max())
    L, H = hi - lo, float(y.max())
    mid = (a > lo + 0.3 * L) & (a < hi - 0.3 * L)
    back = float(np.percentile(y[mid], 97)) if mid.sum() > 20 else 0.6 * H
    score, mass = {}, {}
    for sign, end, tip in ((1, a > hi - 0.25 * L, a > hi - 0.12 * L),
                           (-1, a < lo + 0.25 * L, a < lo + 0.12 * L)):
        score[sign] = float(np.clip(y[end] - back - 0.02 * H, 0, None).sum())
        mass[sign] = float((tip & (y > 0.4 * H)).sum())
    notes = []
    if max(score.values()) > 1e-4 * len(pts) * H:
        sign = max(score, key=score.get)
        if score[sign] < 1.5 * score[-sign]:
            notes.append("head end: unsure (both ends rise above the back); pass head= to fix")
    else:
        sign = max(mass, key=mass.get)
        notes.append("head end: nothing above the back, guessed the heavier end; pass head= to fix")
    name = ("+" if sign > 0 else "-") + ("x" if ax == 0 else "z")
    return HEADS[name], name, notes


# --------------------------------------------------------------------------- skeleton

def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _two_means(xs: np.ndarray) -> tuple[float, float, float, bool]:
    """Split values in two groups: (low mean, high mean, split, clearly separate?)."""
    a, b = np.percentile(xs, 15), np.percentile(xs, 85)
    for _ in range(30):
        m = (a + b) / 2
        lo, hi = xs[xs <= m], xs[xs > m]
        if not len(lo) or not len(hi):
            break
        a, b = lo.mean(), hi.mean()
    m = (a + b) / 2
    spread = max(b - a, 1e-9)
    near = np.abs(xs - m) < 0.15 * spread
    density_mid = near.mean() / 0.3
    density_all = 1.0 / max(np.ptp(xs) / spread, 1e-9)
    return float(a), float(b), float(m), bool(density_mid < 0.35 * density_all)


def find_quadruped_skeleton(pts: np.ndarray, head: str = "auto"
                            ) -> tuple[QuadSkeleton, _Grid, np.ndarray]:
    """Joints of a standing four-legged animal from its surface points (up +y, on y = 0).
    The points are turned so the head faces +z first; the skeleton is in that frame (yaw)."""
    yaw, head_end, notes = facing_yaw(pts, head)
    pts = _turn(pts, yaw)
    # re-centre like _load does after a turn
    lo, hi = pts.min(0), pts.max(0)
    pts = pts - np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])
    H = float(pts[:, 1].max())
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z0, z1 = float(z.min()), float(z.max())
    L = z1 - z0
    side = np.stack([z, y, x], 1)              # side view: horizontal = z, depth = x
    grid = _Grid(side, rows=160)
    m = grid.mask
    rows = range(grid.pad + grid.res - 1, grid.pad - 1, -1)   # bottom row first

    def top_at(zz):                            # top of the silhouette at this z
        _, c = grid.rc(zz, 0)
        filled = np.nonzero(m[:, c])[0]
        return grid.xy(int(filled.min()), c)[1] if len(filled) else 0.8 * H

    belly = None
    for r in rows:
        yy = grid.xy(r, 0)[1]
        if yy < 0.08 * H:
            continue
        if yy > 0.8 * H:
            break
        longest = max((b - a for a, b in _runs(m[r])), default=0) * grid.cell
        if longest >= 0.45 * L:
            belly = yy
            break
    legs_ok = belly is not None
    if not legs_ok:
        belly = 0.45 * H
        notes.append("legs: no belly line (lying down?), standard dog proportions")

    # a tail hanging down behind the hind legs reaches below the belly but not the ground:
    # it is not a leg (else the wolf's tail walks like its right hind leg)
    feet = pts[y < 0.05 * H]
    hanging = np.zeros(len(pts), bool)
    if legs_ok and len(feet) > 20:
        zf = float(np.percentile(feet[:, 2], 1))
        hanging = (y < 0.9 * belly) & (z < zf - 0.03 * L) & (y > 0.04 * H)
        if hanging.sum() < max(30, 0.002 * len(pts)):
            hanging[:] = False
    low = pts[(y < 0.6 * belly) & ~hanging]
    pair_z = None
    if legs_ok and len(low) > 40:
        bins = np.arange(z0, z1 + grid.cell, grid.cell)
        hist, edges = np.histogram(low[:, 2], bins)
        nz = np.nonzero(hist)[0]
        best, run, start = (0, 0), 0, 0
        for i in range(nz.min(), nz.max() + 1):
            if hist[i] == 0:
                if run == 0:
                    start = i
                run += 1
                if run > best[0]:
                    best = (run, start)
            else:
                run = 0
        if best[0] >= 2:
            pair_z = float(edges[best[1]] + best[0] * grid.cell / 2)
    if pair_z is None:
        if legs_ok:
            notes.append("legs: front and back pairs not apart, standard dog proportions")
        legs_ok = False
        pair_z = z0 + 0.45 * L

    body_pts = pts[y > 0.9 * belly]
    cx = float(np.median(body_pts[:, 0])) if len(body_pts) else 0.0
    half = float(np.percentile(np.abs(body_pts[:, 0] - cx), 90)) if len(body_pts) else 0.15 * H
    joints: dict[str, np.ndarray] = {}
    tips: dict[str, np.ndarray] = {}
    split_x: dict[str, float] = {}
    leg_pts = pts[(y < 0.9 * belly) & ~hanging]
    for pair in ("front", "back"):
        sel = leg_pts[(leg_pts[:, 2] > pair_z) == (pair == "front")] if legs_ok else leg_pts[:0]
        low_sel = sel[sel[:, 1] < 0.6 * belly]
        if legs_ok and len(low_sel) > 20:
            xr, xl, split, apart = _two_means(low_sel[:, 0])
            if not apart:
                notes.append(f"{pair} legs: touching in depth, split down the middle")
        else:
            split = cx
            xr, xl = cx - 0.55 * half, cx + 0.55 * half
        split_x[pair] = split
        for s, xs in (("l", xl), ("r", xr)):
            leg = f"{pair}_{s}"
            mine = sel[(sel[:, 0] > split) == (s == "l")] if len(sel) else sel
            if len(mine) < 10:
                zz = (z0 + 0.72 * L) if pair == "front" else (z0 + 0.16 * L)
                mine = None
            else:
                zz = float(np.median(mine[:, 2]))
                xs = float(np.median(mine[:, 0]))

            def z_at(h, mine=mine, zz=zz):
                if mine is None:
                    return zz
                band = mine[np.abs(mine[:, 1] - h) < 0.05 * H]
                return float(np.median(band[:, 2])) if len(band) > 3 else zz
            top = belly + 0.4 * (top_at(z_at(0.85 * belly)) - belly)
            # elbow / stifle just under the body, wrist / hock low: thigh ~ shin, like a dog
            mid = (0.8 if pair == "front" else 0.82) * belly
            paw = max(0.04 * H, (0.22 if pair == "front" else 0.3) * belly)
            joints[f"{leg}_upper"] = np.array([xs, top, z_at(0.85 * belly)])
            joints[f"{leg}_lower"] = np.array([xs, mid, z_at(mid)])
            joints[f"{leg}_paw"] = np.array([xs, paw, z_at(paw)])
            if mine is not None and (mine[:, 1] < paw).sum() > 3:
                toe_z = float(np.percentile(mine[mine[:, 1] < paw, 2], 95))
            else:
                toe_z = z_at(paw) + 0.04 * H
            tips[f"{leg}_toe"] = np.array([xs, 0.01 * H, max(toe_z, z_at(paw) + 0.02 * H)])

    def mid_y(zz, k=0.55):
        return belly + k * (top_at(zz) - belly)
    hz = float(np.mean([joints[f"back_{s}_upper"][2] for s in "lr"]))
    fz = float(np.mean([joints[f"front_{s}_upper"][2] for s in "lr"]))
    for name, zz in (("hips", hz), ("spine", (hz + fz) / 2), ("chest", fz)):
        yy = mid_y(zz)
        joints[name] = np.array([grid.depth(zz, yy), yy, zz])

    # head: the mass above the back at the front
    chest = joints["chest"]
    front = pts[(z > fz) & (y > belly)]
    if len(front) > 20:
        y_top = float(front[:, 1].max())
        head_pts = front[front[:, 1] > y_top - 0.3 * (y_top - belly)]
        hc = head_pts.mean(0)
        if hc[2] < fz + 0.03 * L or y_top < mid_y(fz):
            hc = np.array([chest[0], max(y_top, chest[1] + 0.15 * H), z1 - 0.1 * L])
            notes.append("head: not clear of the body, placed at the front")
        snout = head_pts[int(np.argmax(head_pts[:, 2]))]
        tips["head_tip"] = np.array([hc[0], snout[1], max(snout[2], hc[2] + 0.02 * H)])
        tips["head_top"] = np.array([hc[0], y_top, hc[2]])
    else:
        hc = np.array([chest[0], 0.85 * H, z1 - 0.1 * L])
        tips["head_tip"], tips["head_top"] = np.array([hc[0], hc[1], z1]), np.array([hc[0], H, hc[2]])
        notes.append("head: not found, standard proportions")
    joints["neck"] = chest + 0.4 * (hc - chest)
    joints["head"] = chest + 0.78 * (hc - chest)

    # tail: thin silhouette columns at the back
    hips = joints["hips"]
    thick = m[grid.pad:grid.pad + grid.res].sum(0)
    r_belly, _ = grid.rc(0, belly)
    _, c_h = grid.rc(hz, 0)
    _, c_0 = grid.rc(z0, 0)
    ref = max(1, int(m[:r_belly, c_h].sum()))          # body only, above the belly
    c = c_0
    while c < c_h and 0 < thick[c] < 0.3 * ref:
        c += 1
    has_tail = (c - c_0) * grid.cell >= 0.06 * L
    if hanging.any():                                  # the hanging tail: root, middle, tip
        hp = pts[hanging]
        top_y = float(hp[:, 1].max())
        base = np.array([float(np.median(hp[:, 0])), top_y + 0.5 * max(0.0, hips[1] - top_y),
                         float(np.percentile(hp[:, 2], 90))])
        tip = hp[np.argmin(hp[:, 1] + 0.3 * (hp[:, 2] - hp[:, 2].min()))]
        band = np.abs(hp[:, 1] - (top_y + tip[1]) / 2) < 0.05 * H
        midp = np.median(hp[band], 0) if band.sum() > 3 else (base + tip) / 2
        joints["tail1"], joints["tail2"], tips["tail_tip"] = base, midp, tip
        has_tail = True
        notes.append("tail: hanging down behind the hind legs")
    elif has_tail:
        base_z = grid.xy(0, c)[0]

        def centre(z_lo, z_hi):
            sel = pts[(z >= z_lo) & (z <= z_hi)]
            return sel.mean(0) if len(sel) else None
        base = centre(base_z - 2 * grid.cell, base_z)
        tip = centre(z0, z0 + 2 * grid.cell)
        midp = centre((base_z + z0) / 2 - grid.cell, (base_z + z0) / 2 + grid.cell)
        if base is None or tip is None or midp is None:
            has_tail = False
        else:
            joints["tail1"], joints["tail2"], tips["tail_tip"] = base, midp, tip
    rear = body_pts[body_pts[:, 2] < hz] if len(body_pts) else body_pts
    rump_z = float(np.percentile(rear[:, 2], 2)) if len(rear) > 10 else z0
    if has_tail:
        rump_z = max(rump_z, float(joints["tail1"][2]))
    tips["rump"] = np.array([hips[0], hips[1], rump_z + 0.04 * L])
    if not has_tail:
        t1 = np.array([hips[0], hips[1] + 0.25 * (top_at(rump_z + 0.05 * L) - hips[1]), rump_z + 0.03 * L])
        joints["tail1"], joints["tail2"] = t1, t1 + [0, 0.02 * H, -0.03 * L]
        tips["tail_tip"] = t1 + [0, 0.04 * H, -0.06 * L]
    sample = body_pts[np.random.default_rng(0).choice(len(body_pts), min(3000, len(body_pts)),
                                                      replace=False)] if len(body_pts) else pts[:1]
    sk = QuadSkeleton(joints, tips, H, notes, yaw, head_end, L, belly, (fz + hz) / 2, split_x,
                      has_tail, sample)
    return sk, grid, m


# --------------------------------------------------------------------------- skinning

def _bones(sk: QuadSkeleton):
    j, t = sk.joints, sk.tips
    segs = [("hips", j["hips"], j["spine"]), ("hips", j["hips"], t["rump"]),
            ("spine", j["spine"], j["chest"]), ("chest", j["chest"], j["neck"]),
            ("neck", j["neck"], j["head"]), ("head", j["head"], t["head_tip"]),
            ("head", j["head"], t["head_top"])]
    for leg in LEGS:
        root = "chest" if leg.startswith("front") else "hips"
        segs += [(root, j[root], j[f"{leg}_upper"]),
                 (f"{leg}_upper", j[f"{leg}_upper"], j[f"{leg}_lower"]),
                 (f"{leg}_lower", j[f"{leg}_lower"], j[f"{leg}_paw"]),
                 (f"{leg}_paw", j[f"{leg}_paw"], t[f"{leg}_toe"])]
    if sk.has_tail:
        segs += [("tail1", j["tail1"], j["tail2"]), ("tail2", j["tail2"], t["tail_tip"])]
    return segs


def skin_weights(vertices: np.ndarray, sk: QuadSkeleton, influences: int = 4,
                 power: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """(joint indices, weights) per vertex: nearest bones by distance to the bone segment,
    legs only on their own side and pair, below the body."""
    v = vertices
    dist = np.full((len(v), len(NAMES)), np.inf)
    for owner, a, b in _bones(sk):
        ab = b - a
        t = np.clip(((v - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        d = np.linalg.norm(v - (a + t[:, None] * ab), axis=1)
        k = NAMES.index(owner)
        dist[:, k] = np.minimum(dist[:, k], d)
    front_half = v[:, 2] > sk.pair_z
    for leg in LEGS:
        pair, s = leg.split("_")
        mine = (front_half == (pair == "front")) & ((v[:, 0] > sk.split_x[pair]) == (s == "l"))
        mine &= v[:, 1] < sk.joints[f"{leg}_upper"][1] + 0.02 * sk.height
        for p in PARTS:
            dist[~mine, NAMES.index(f"{leg}_{p}")] = np.inf
    if sk.has_tail:   # the tail doesn't take the rump
        behind = v[:, 2] < sk.joints["tail1"][2] + 0.02 * sk.length
        for n in ("tail1", "tail2"):
            dist[~behind, NAMES.index(n)] = np.inf
    # under the body: legs first (body bones count 4x further away)
    leg_cols = [NAMES.index(f"{leg}_{p}") for leg in LEGS for p in PARTS]
    body_cols = [NAMES.index(n) for n in ("hips", "spine", "chest", "neck", "head")]
    under = v[:, 1] < 0.85 * sk.belly
    near_leg = np.isfinite(dist[:, leg_cols]).any(1)
    dist[np.ix_(under & near_leg, body_cols)] *= 4.0
    w = np.where(np.isfinite(dist), 1.0 / (dist / sk.height + 0.01) ** power, 0.0)
    idx = np.argsort(-w, axis=1)[:, :influences]
    top = np.take_along_axis(w, idx, 1)
    top /= np.maximum(top.sum(1, keepdims=True), 1e-12)
    return idx.astype(np.uint8), top.astype(np.float32)


# --------------------------------------------------------------------------- posing

_I = np.array([0.0, 0.0, 0.0, 1.0])


def _rx(deg):
    return _q("x", deg)


def _conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _rot(q, v):
    """Rotate vector(s) v (..., 3) by quaternion q."""
    u, w = np.asarray(q[:3]), q[3]
    v = np.asarray(v, np.float64)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _ang(v) -> float:
    """Side-plane angle of a bone: 0 = pointing down, + = pointing forward (+z), degrees."""
    return math.degrees(math.atan2(v[2], -v[1]))


def _q_of(spec) -> np.ndarray:
    q = _I.copy()
    for axis, deg in spec:
        q = _qmul(q, _q(axis, deg))
    return q


class _Poser:
    """Procedural quadruped poses: body rotations + IK legs towards foot targets."""

    def __init__(self, sk: QuadSkeleton):
        self.sk, self.H, j = sk, sk.height, sk.joints
        self.j = j
        self.off = {n: j[n] - (j[p] if p else 0) for n, p in JOINTS}
        self.legs = {}
        for leg in LEGS:
            t, m, p = (j[f"{leg}_{k}"] for k in PARTS)
            toe = sk.tips[f"{leg}_toe"]
            l1 = float(np.hypot(*(m - t)[1:]))
            l2 = float(np.hypot(*(p - m)[1:]))
            a1, a2, a3 = _ang(m - t), _ang(p - m), _ang(toe - p)
            line = _ang(p - t)
            bend = a1 - line
            if abs(bend) < 3:
                bend = -1.0 if leg.startswith("front") else 1.0   # elbow back, knee forward
            self.legs[leg] = dict(l1=l1, l2=l2, a1=a1, a2=a2, a3=a3, bend=math.copysign(1, bend))
        self.leg_len = float(np.mean([v["l1"] + v["l2"] for v in self.legs.values()]))

    # -- kinematics
    def fk(self, local: dict, root_off) -> tuple[dict, dict]:
        wpos, wrot = {}, {}
        for n, p in JOINTS:
            if p is None:
                wrot[n], wpos[n] = local[n], self.j[n] + root_off
            else:
                wrot[n] = _qmul(wrot[p], local[n])
                wpos[n] = wpos[p] + _rot(wrot[p], self.off[n])
        return wpos, wrot

    def ik(self, leg, root, target):
        """World side-plane angles (upper, lower) reaching target (y, z) from root (y, z)."""
        L = self.legs[leg]
        dy, dz = target[0] - root[0], target[1] - root[1]
        d = math.hypot(dy, dz)
        d = min(max(d, abs(L["l1"] - L["l2"]) + 1e-6, 1e-6), (L["l1"] + L["l2"]) * 0.9999)
        line = math.degrees(math.atan2(dz, -dy))
        cos_a = (L["l1"] ** 2 + d * d - L["l2"] ** 2) / (2 * L["l1"] * d)
        a = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
        a1 = line + L["bend"] * a
        mid = (root[0] - L["l1"] * math.cos(math.radians(a1)),
               root[1] + L["l1"] * math.sin(math.radians(a1)))
        a2 = math.degrees(math.atan2(target[1] - mid[1], -(target[0] - mid[0])))
        return a1, a2

    def pose(self, body: dict, root_off, feet: dict, fixed: dict | None = None):
        """body: joint -> quaternion (non-leg joints); feet: leg -> (y, z, paw world degrees)
        target of the paw joint; fixed: leg joint -> local quaternion (no IK for that leg)."""
        local = {n: body.get(n, _I) for n in NAMES}
        fixed = fixed or {}
        for n, q in fixed.items():
            local[n] = q
        wpos, wrot = self.fk(local, np.asarray(root_off, np.float64))
        for leg, (ty, tz, paw) in feet.items():
            L = self.legs[leg]
            up = f"{leg}_upper"
            root = wpos[up]
            a1, a2 = self.ik(leg, (root[1], root[2]), (ty, tz))
            th1, th2 = L["a1"] - a1, L["a2"] - a2
            local[up] = _qmul(_conj(wrot[PARENT[up]]), _rx(th1))
            local[f"{leg}_lower"] = _rx(th2 - th1)
            local[f"{leg}_paw"] = _rx(paw - th2)
        return local

    def ground(self, local, root_off) -> float:
        """Lowest point of the posed body sample, toes and paws."""
        wpos, wrot = self.fk(local, np.asarray(root_off, np.float64))
        hips = self.j["hips"]
        body = _rot(wrot["hips"], self.sk.body - hips) + wpos["hips"]
        low = [float(body[:, 1].min())]
        for leg in LEGS:
            p = f"{leg}_paw"
            toe = wpos[p] + _rot(wrot[p], self.sk.tips[f"{leg}_toe"] - self.j[p])
            low += [float(toe[1]), float(wpos[p][1]) - 0.03 * self.H, float(wpos[f"{leg}_lower"][1])]
        return min(low)

    # -- gaits
    def _drop(self, stride):
        """How far the body must come down so stance feet reach stride ahead and behind."""
        need = 0.0
        for leg in LEGS:
            L = self.legs[leg]
            t, p = self.j[f"{leg}_upper"], self.j[f"{leg}_paw"]
            dy, e = t[1] - p[1], p[2] - t[2]
            reach = 0.985 * (L["l1"] + L["l2"])
            for du in (e - stride, e + stride):
                if abs(du) < reach:
                    need = max(need, dy - math.sqrt(reach ** 2 - du ** 2))
        return max(0.0, need)

    def feet(self, p, phases, duty, stride, lift, flex):
        out = {}
        for leg, ph in phases.items():
            q = (p - ph) % 1.0
            rest = self.j[f"{leg}_paw"]
            if q < duty:
                s = q / duty
                dz, dy, paw = stride * (1 - 2 * s), 0.0, 0.0
            else:
                s = (q - duty) / (1 - duty)
                dz = stride * (-1 + 2 * _smooth(s))
                dy = lift * math.sin(math.pi * s)
                paw = flex[0 if leg.startswith("front") else 1] * math.sin(math.pi * s) ** 1.5
            out[leg] = (rest[1] + dy, rest[2] + dz, paw)
        return out

    def planted(self):
        return {leg: (self.j[f"{leg}_paw"][1], self.j[f"{leg}_paw"][2], 0.0) for leg in LEGS}

    # -- actions
    def quats(self, action: str, p: float):
        Lg = self.leg_len
        c2 = math.cos(2 * math.pi * p)
        s2 = math.sin(2 * math.pi * p)
        if action == "idle":
            b = (1 - c2) / 2
            body = {"chest": _rx(-1.2 * b), "neck": _q_of([("y", 6 * s2), ("x", 2 * b)]),
                    "head": _rx(-2 * b), "tail1": _q_of([("y", 14 * s2)]),
                    "tail2": _q_of([("y", 10 * math.sin(2 * math.pi * p - 0.8))])}
            off = np.array([0, -0.012 * Lg * b, 0])
            return self.pose(body, off, self.planted()), off
        if action == "walk":
            stride, lift = 0.2 * Lg, 0.16 * Lg
            phases = {"back_l": 0.0, "front_l": 0.25, "back_r": 0.5, "front_r": 0.75}
            feet = self.feet(p, phases, 0.65, stride, lift, (55, 25))
            bob = 0.012 * Lg * (1 + math.cos(4 * math.pi * p)) / 2
            body = {"hips": _q_of([("y", 3 * s2), ("z", 2 * s2)]),
                    "chest": _q_of([("y", -5 * s2), ("z", -2 * s2)]),
                    "neck": _rx(3 * math.cos(4 * math.pi * p + 0.6)), "head": _rx(-2 * math.cos(4 * math.pi * p + 0.6)),
                    "tail1": _q_of([("y", 14 * s2)]), "tail2": _q_of([("y", 10 * math.sin(2 * math.pi * p - 0.8))])}
            off = np.array([0, -self._drop(stride) - bob, 0])
            return self.pose(body, off, feet), off
        if action == "run":
            stride, lift = 0.36 * Lg, 0.26 * Lg
            phases = {"back_l": 0.0, "back_r": 0.1, "front_l": 0.48, "front_r": 0.58}
            feet = self.feet(p, phases, 0.4, stride, lift, (85, 45))
            flexion = math.sin(2 * math.pi * (p - 0.3))
            body = {"hips": _rx(-5 * flexion), "spine": _rx(4 * flexion), "chest": _rx(4 * flexion),
                    "neck": _rx(-6 * flexion), "head": _rx(-3 * flexion),
                    "tail1": _q_of([("x", -10 + 8 * flexion)]), "tail2": _rx(6 * flexion)}
            drop = self._drop(stride)
            off = np.array([0, -drop - 0.02 * Lg * (1 + math.cos(4 * math.pi * (p - 0.05))) / 2, 0])
            return self.pose(body, off, feet), off
        if action == "happy":
            bounce = (1 - math.cos(4 * math.pi * p)) / 2
            body = {"hips": _q_of([("y", 4 * s2)]), "chest": _rx(-3 * bounce),
                    "neck": _q_of([("x", 7 * math.sin(4 * math.pi * p)), ("y", -5 * s2)]),
                    "head": _rx(-4 * math.sin(4 * math.pi * p)),
                    "tail1": _q_of([("y", 35 * math.sin(4 * math.pi * p)), ("x", -15)]),
                    "tail2": _q_of([("y", 25 * math.sin(4 * math.pi * p - 1.0))])}
            off = np.array([0, -0.04 * Lg * bounce, 0])
            return self.pose(body, off, self.planted()), off
        if action == "sit":
            return self._sit(p)
        if action == "lie_down":
            return self._lie(p)
        if action == "death":
            return self._death(p)
        from .motions import quad_pose        # bark, howl, jump, eat
        return quad_pose(self, action, p)

    def _sit_target(self):
        if hasattr(self, "_sit_cache"):
            return self._sit_cache
        hips = self.j["hips"]
        body = self.sk.body - hips
        best = None
        for deg in np.arange(0, -56, -1.0):
            q = _rx(deg)
            d = float((_rot(q, body)[:, 1] + hips[1]).min()) - 0.01 * self.H
            worst = 1.0
            for s in "lr":
                leg = f"front_{s}"
                t = _rot(q, self.j[f"{leg}_upper"] - hips) + hips - [0, d, 0]
                p = self.j[f"{leg}_paw"]
                reach = self.legs[leg]["l1"] + self.legs[leg]["l2"]
                worst = min(worst, math.hypot(t[1] - p[1], t[2] - p[2]) / reach)
            if best is None or worst > best[2] or worst >= 0.93:
                best = (float(deg), d, worst)
            if worst >= 0.93:
                break
        self._sit_cache = best[:2]
        return self._sit_cache

    def _sit(self, p):
        deg, d = self._sit_target()
        u = _smooth(min(1.0, p / 0.7))
        body = {"hips": _rx(deg * u), "neck": _rx(-0.55 * deg * u), "head": _rx(-0.25 * deg * u),
                "tail1": _rx(0.6 * deg * u)}
        off = np.array([0, -d * u, 0])
        feet = self.planted()
        local = {n: body.get(n, _I) for n in NAMES}
        wpos, _ = self.fk(local, off)
        for s in "lr":
            leg = f"back_{s}"
            rest = self.j[f"{leg}_paw"]
            root = wpos[f"{leg}_upper"]
            ty = max(0.035 * self.H, 0.5 * rest[1])
            tz = root[2] - 0.05 * self.leg_len
            paw = self.legs[leg]["a3"] - 88
            lift = 0.1 * self.leg_len * math.sin(math.pi * u)
            feet[leg] = (rest[1] + (ty - rest[1]) * u + lift, rest[2] + (tz - rest[2]) * u, paw * u)
        return self.pose(body, off, feet), off

    def _lie(self, p):
        lie = self.sk.belly - 0.02 * self.H
        u = _smooth(min(1.0, p / 0.8))
        body = {"neck": _rx(6 * u), "head": _rx(-4 * u), "tail1": _rx(-6 * u)}
        off = np.array([0, -lie * u, 0])
        local = {n: body.get(n, _I) for n in NAMES}
        wpos, _ = self.fk(local, off)
        feet = {}
        for leg in LEGS:
            rest = self.j[f"{leg}_paw"]
            root = wpos[f"{leg}_upper"]
            L = self.legs[leg]
            ty = rest[1]
            if leg.startswith("front"):
                reach = 0.96 * (L["l1"] + L["l2"])
                tz = root[2] + math.sqrt(max(reach ** 2 - (root[1] - ty) ** 2, 0.0))
                paw = L["a3"] - 85
                lift = 0.12 * self.leg_len * math.sin(math.pi * min(1.0, u * 1.2))
            else:
                ty = max(0.035 * self.H, 0.5 * rest[1])
                tz = root[2] - 0.1 * self.leg_len
                paw = L["a3"] - 88
                lift = 0.05 * self.leg_len * math.sin(math.pi * u)
            feet[leg] = (rest[1] + (ty - rest[1]) * u + lift, rest[2] + (tz - rest[2]) * u, paw * u)
        return self.pose(body, off, feet), off

    def _death(self, p):
        u = _smooth(min(1.0, p / 0.75))
        roll = 88 * u
        body = {"hips": _q_of([("z", roll)]), "neck": _rx(12 * u), "head": _rx(8 * u),
                "tail1": _rx(10 * u)}
        fixed = {}
        for leg in LEGS:
            front = leg.startswith("front")
            fixed[f"{leg}_upper"] = _rx((-12 if front else 10) * u)
            fixed[f"{leg}_lower"] = _rx((20 if front else -18) * u)
            fixed[f"{leg}_paw"] = _rx(25 * u)
        local = self.pose(body, np.zeros(3), {}, fixed)
        off = np.array([0.0, 0.0, 0.0])
        off[1] = -(self.ground(local, off) - 0.005 * self.H)
        return local, off


# --------------------------------------------------------------------------- glTF out + preview

def all_actions() -> tuple[str, ...]:
    """Every animal clip: the built-in seven and motions.py's bark, howl, jump, eat."""
    from .motions import QUAD
    return ACTIONS + tuple(QUAD)


def clip_info(motion: str) -> tuple[float, bool]:
    if motion in DURATION:
        return DURATION[motion], LOOPS[motion]
    from .motions import QUAD
    return QUAD[motion]


def quad_clips(sk: QuadSkeleton, actions, keys: int = 25) -> list:
    """[(clip name, times, poses, loop)]; 'name=motion' renames like the humanoid rig."""
    from .motions import parse_actions
    poser = _Poser(sk)
    clips = []
    for name, motion in parse_actions(actions, all_actions()):
        dur, loop = clip_info(motion)
        times = np.linspace(0, dur, keys)
        clips.append((name, times, [poser.quats(motion, t / dur) for t in times], loop))
    return clips


def quad_extras(sk: QuadSkeleton, body_points: int = 1500) -> dict:
    from .autorig import skeleton_extras
    body = sk.body if sk.body is not None else np.zeros((0, 3))
    if len(body) > body_points:
        body = body[np.linspace(0, len(body) - 1, body_points).astype(int)]
    return skeleton_extras(sk, "quadruped", yaw=float(sk.yaw), head=sk.head,
                           length=float(sk.length), belly=float(sk.belly),
                           pair_z=float(sk.pair_z),
                           split_x={k: float(v) for k, v in sk.split_x.items()},
                           has_tail=bool(sk.has_tail),
                           body=np.round(body, 5).tolist())


def write_quadruped(parts, sk: QuadSkeleton, out_glb, actions=ACTIONS, keys: int = 25) -> Path:
    clips = quad_clips(sk, actions, keys)
    return write_skinned(parts, JOINTS, sk.joints, lambda v: skin_weights(v, sk), clips, out_glb,
                         extras=quad_extras(sk))


def rig_preview(grid: _Grid, sk: QuadSkeleton, scale: int = 2) -> np.ndarray:
    """Side silhouette (grey, head to the right) with the joints and bones found."""
    m = grid.mask
    img = np.zeros(m.shape + (4,), np.uint8)
    img[m] = (150, 156, 170, 255)
    im = Image.fromarray(img).resize((m.shape[1] * scale, m.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def px(p):
        r, c = grid.rc(p[2], p[1])
        return ((c + 0.5) * scale, (r + 0.5) * scale)
    colours = {"l": (60, 200, 90), "r": (230, 80, 70)}
    for owner, a, b in _bones(sk):
        side = owner.split("_")[1] if owner.startswith(("front_", "back_")) else None
        d.line([px(a), px(b)], fill=colours.get(side, (250, 210, 60)), width=max(1, scale))
    for p in sk.joints.values():
        x, y = px(p)
        d.ellipse([x - 2 * scale, y - 2 * scale, x + 2 * scale, y + 2 * scale], fill=(255, 255, 255),
                  outline=(20, 20, 30))
    return np.asarray(im)


def auto_rig_quadruped(model, out_glb=None, *, actions=ACTIONS, head: str = "auto", up: str = "y",
                       yaw: float = 0.0, points: int = 400_000) -> QuadRigResult:
    """A standing four-legged animal -> rigged, animated .glb (default <name>.rigged.glb),
    turned so its head faces +z. head: auto, or which end of the model is the head
    (+x, -x, +z, -z after `up`/`yaw`). actions: any of all_actions() (also bark, howl, jump,
    eat from motions.py), 'name=motion' to rename."""
    from .motions import parse_actions
    parse_actions(actions, all_actions())
    model = Path(model)
    out = Path(out_glb) if out_glb else model.with_name(model.stem + ".rigged.glb")
    parts = _load(model, up, yaw)
    pts = _surface_points(parts, points)
    align = body_axis_yaw(pts)                 # a diagonal body (three-quarter picture) first
    if align:
        parts = _load(model, up, yaw + align)
        pts = _surface_points(parts, points)
    turn, name, notes = facing_yaw(pts, head)
    if align:
        notes = [f"body lay {align:+.0f} degrees off the axes: turned it straight"] + notes
    if turn:
        parts = _load(model, up, yaw + align + turn)
        pts = _surface_points(parts, points)
    sk, grid, _ = find_quadruped_skeleton(pts, "+z")
    sk.yaw, sk.head = turn + align, name
    sk.notes = notes + sk.notes
    write_quadruped(parts, sk, out, actions)
    return QuadRigResult(out, sk, rig_preview(grid, sk),
                         [n for n, _ in parse_actions(actions, all_actions())])

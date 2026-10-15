"""Secondary motion: capes, hair and tails swing and settle (spring bones baked into every clip).

    add_springs("rigged.glb")                      # finds the cape / hair / tail itself
    add_springs("rigged.glb", parts=["cape"], stiffness=0.3)

1. What dangles: every vertex's distance from the body's core (capsules around the bones,
   each as thick as the body around it measured on its front side, where capes don't hang) says
   how loose it is. Loose geometry below the neck and behind / beside the torso is a cape, loose
   geometry around the back of the head is hair; a four-legged rig's tail is what its tail
   bones carry. parts="auto" takes what it finds, or a list of "cape", "hair", "tail".
2. Each part gets a short chain of extra bones (3 by default; a cape wrapping round the sides
   gets three chains: left, back, right) hanging from the bone it is attached to (chest, head,
   tail1). Its vertices are skinned to the chain with smooth falloffs: along the chain, across
   neighbouring chains, and from "loose" to "body" near the seam (no tearing at the shoulders).
3. Simulation per clip: each chain point is pulled towards where the chain would be if it were
   rigid (stiffness), damped, pushed by gravity (relative to the rest shape, so a hanging cape
   doesn't sag at rest but falls when the body tips over), by the air of travelling (clips that
   walk or run: the cape trails), kept at its bone lengths, out of the body's capsules and above
   the ground. Loops are simulated over three cycles and the last one is kept with its seam
   blended, so they still loop; one-shots start from a settled cape.
4. The result is baked into the clip keys (rotation channels of the spring bones), so sprite
   exports, Godot and Blender all see it. Running add_springs again on a file that has springs
   only re-simulates (new clips added later get their swing).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mocap import qarc, qconj, qfrom_matrix, qmul, qnorm, qrot

PREFIX = "spring_"
PART_WIND = {"cape": 1.0, "hair": 0.6, "tail": 0.3}     # how much the air of travelling lifts it
MAX_SWING = 55.0                       # degrees a spring bone may swing from where it hangs
COS_MAX, SIN_MAX = math.cos(math.radians(MAX_SWING)), math.sin(math.radians(MAX_SWING))
DEFAULT_SPEED = {"walk": 0.8, "run": 2.0, "jog": 1.4, "sprint": 2.6}   # body heights / second


@dataclass
class Chain:
    name: str                 # cape, cape_l, cape_r, hair, tail
    parent: str               # the existing joint it hangs from
    points: np.ndarray        # (n + 1, 3) rest positions: [0] anchor ... [n] tip

    @property
    def joints(self) -> list[str]:
        return [f"{PREFIX}{self.name}_{i}" for i in range(len(self.points) - 1)]


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _seg_dist(v, a, b):
    ab = b - a
    t = np.clip(((v - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(v - (a + t[:, None] * ab), axis=1)


# --------------------------------------------------------------------------- what dangles

def _core(kind, sk):
    """Body capsules [(name, owner joint, a, b)] in the rest pose."""
    j, t = sk.joints, sk.tips
    if kind == "humanoid":
        pelvis = (j["thigh_l"] + j["thigh_r"]) / 2 + np.array([0, 0.04 * sk.height, 0])
        caps = [("pelvis", "hips", pelvis, j["spine"]), ("spine", "spine", j["spine"], j["chest"]),
                ("chest", "chest", j["chest"], j["neck"]), ("neck", "neck", j["neck"], j["head"]),
                ("head", "head", j["head"], t["head_top"])]
        for s in "lr":
            caps += [(f"upper_arm_{s}", f"upper_arm_{s}", j[f"upper_arm_{s}"], j[f"forearm_{s}"]),
                     (f"forearm_{s}", f"forearm_{s}", j[f"forearm_{s}"], j[f"hand_{s}"]),
                     (f"thigh_{s}", f"thigh_{s}", j[f"thigh_{s}"], j[f"shin_{s}"]),
                     (f"shin_{s}", f"shin_{s}", j[f"shin_{s}"], j[f"foot_{s}"])]
        return caps
    caps = [("hips", "hips", j["hips"], j["spine"]), ("spine", "spine", j["spine"], j["chest"]),
            ("chest", "chest", j["chest"], j["neck"]), ("neck", "neck", j["neck"], j["head"])]
    from .quadruped import LEGS
    for leg in LEGS:
        caps += [(f"{leg}_upper", f"{leg}_upper", j[f"{leg}_upper"], j[f"{leg}_lower"]),
                 (f"{leg}_lower", f"{leg}_lower", j[f"{leg}_lower"], j[f"{leg}_paw"])]
    return caps


def _radii(v, caps, pct: float = 35):
    """Per capsule: the body's thickness there (a low percentile of the distances of the
    vertices nearest to it, so cloth hanging off it doesn't count)."""
    d = np.stack([_seg_dist(v, a, b) for _, _, a, b in caps], 1)
    near = np.argmin(d, 1)
    out = []
    for k, (_, _, a, b) in enumerate(caps):
        sel = near == k
        front = sel & (v[:, 2] >= min(a[2], b[2]) - 0.01)     # capes hang behind: measure in front
        if front.sum() >= 8:
            sel = front
        out.append(float(np.percentile(d[sel, k], pct)) if sel.sum() >= 4 else 0.03)
    return np.array(out), d


def _front_depth(v, H, cell=0.02, spread=0.05):
    """How far behind the body's front each vertex is: the front-most surface in a window
    around its column of the front view (wide enough to reach the legs from between them)."""
    import cv2
    c = cell * H
    col = np.floor((v[:, 0] - v[:, 0].min()) / c).astype(int)
    row = np.floor((v[:, 1] - v[:, 1].min()) / c).astype(int)
    grid = np.full((row.max() + 1, col.max() + 1), -1e9, np.float32)
    np.maximum.at(grid, (row, col), v[:, 2].astype(np.float32))
    k = max(1, int(round(spread / cell)))
    kernel = np.ones((3, 2 * k + 1), np.uint8)
    front = cv2.dilate(grid, kernel)
    have = (front > -1e8).astype(np.float32)
    front = np.where(have > 0, front, 0).astype(np.float32)
    blur = cv2.GaussianBlur(front * have, (0, 0), 1.5) / np.maximum(cv2.GaussianBlur(have, (0, 0), 1.5), 1e-6)
    return blur[row, col] - v[:, 2]                          # smooth: weights must not jump


def vertex_normals(v, faces):
    n = np.zeros_like(v)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def sheet_thickness(v, faces, H, reach=0.06, sample=20000, per_cell=12, seed=0):
    """How thick the solid is under each vertex (x height): the nearest vertex facing the
    other way behind it. A cape or a skirt is a thin sheet; a body, arm or leg is not.
    Found on a sample of vertices and spread to all (smooth_field); reach caps it."""
    nrm = vertex_normals(v, faces)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(v), min(sample, len(v)), replace=False)
    c = reach * H
    lo = v.min(0)
    cell = np.floor((v - lo) / c).astype(np.int64)
    dims = cell.max(0) + 3

    def keys(cc):
        return ((cc[:, 0] + 1) * dims[1] + cc[:, 1] + 1) * dims[2] + cc[:, 2] + 1
    key = keys(cell)
    order = np.lexsort((rng.random(len(v)), key))    # by cell, shuffled inside each cell
    skey = key[order]
    out = np.full(len(pick), reach)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for b0 in range(0, len(pick), 4000):
        pi = pick[b0:b0 + 4000]
        cand = []
        for off in offs:
            k = keys(cell[pi] + np.array(off))
            st = np.searchsorted(skey, k, "left")
            en = np.searchsorted(skey, k, "right")
            for m in range(per_cell):
                j = st + m
                cand.append(np.where(j < en, order[np.minimum(j, len(order) - 1)], -1))
        cand = np.stack(cand, 1)                               # (b, 27 * per_cell)
        cs = np.maximum(cand, 0)
        d = v[cs] - v[pi][:, None, :]
        along = np.einsum("bkj,bj->bk", d, nrm[pi])
        facing = np.einsum("bkj,bj->bk", nrm[cs], nrm[pi])
        ok = (cand >= 0) & (facing < -0.3) & (along < 0)
        dist = np.where(ok, np.linalg.norm(d, axis=2), np.inf)
        out[b0:b0 + len(pi)] = np.minimum(dist.min(1) / H, reach)
    return smooth_field(v[pick], out, H, sigma=0.015, at=v)


def smooth_field(v, values, H, sigma=0.03, cell=0.02, at=None):
    """values (per point of v) blurred in space (sigma x height): a voxel grid, separable
    Gaussian, trilinear lookup at `at` (default: v). Keeps skin-weight blends wide enough not
    to tear the mesh."""
    c = cell * H
    at = v if at is None else at
    lo = np.minimum(v.min(0), at.min(0)) - 3 * sigma * H
    hi = np.maximum(v.max(0), at.max(0))
    idx = (v - lo) / c
    shape = np.ceil((hi - lo) / c).astype(int) + 4
    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    i0 = np.floor(idx).astype(int)
    np.add.at(acc, tuple(i0.T), values)
    np.add.at(cnt, tuple(i0.T), 1.0)
    r = max(1, int(math.ceil(3 * sigma / cell)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x * cell / sigma) ** 2)
    k /= k.sum()
    for ax in range(3):
        acc = np.apply_along_axis(np.convolve, ax, acc, k, "same")
        cnt = np.apply_along_axis(np.convolve, ax, cnt, k, "same")
    field = acc / np.maximum(cnt, 1e-9)
    f = (at - lo) / c - 0.5                             # cell centres
    b = np.floor(f).astype(int)
    u = f - b
    out = np.zeros(len(at))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1])
                     * (u[:, 2] if dz else 1 - u[:, 2]))
                ii = np.clip(b + [dx, dy, dz], 0, np.array(shape) - 1)
                out += w * field[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def looseness(v, kind, sk):
    """(0-1 per vertex: how far it hangs off the body, capsule distances, radii).
    Loose = far from every bone's capsule (side flaps, tails), or, for people, much deeper
    behind the body's front than a torso / leg is thick (a cape on the back)."""
    caps = _core(kind, sk)
    r, d = _radii(v, caps)
    H = sk.height
    excess = (d - r[None, :]).min(1) / H
    lat = _smoothstep((excess - 0.05) / 0.05)
    if kind != "humanoid":
        return lat, d, r
    behind = _front_depth(v, H) / H
    crotch = float(min(sk.joints["thigh_l"][1], sk.joints["thigh_r"][1]))
    thick = 0.08 + 0.05 * _smoothstep((v[:, 1] - crotch + 0.06 * H) / (0.12 * H))   # legs .. torso
    back = _smoothstep((behind - thick - 0.015) / 0.04)
    return np.maximum(lat, back), d, r


def _chain_along(pts, anchor, n, order_key, outer=None):
    """n + 1 points from the anchor through the middle of pts, ordered by order_key (0 at the
    anchor, growing along the part). outer: per point distance from the body; only the outer
    layer of each band counts (a cape's flap, not the tunic panels under it)."""
    lo, hi = float(order_key.min()), float(order_key.max())
    out = [anchor]
    for i in range(1, n + 1):
        at = lo + (hi - lo) * i / n
        band = np.nonzero(np.abs(order_key - at) <= (hi - lo) / (2 * n) + 1e-9)[0]
        if len(band) < 3:
            band = np.argsort(np.abs(order_key - at))[:max(3, len(pts) // (2 * n))]
        if outer is not None and len(band) > 10:
            band = band[outer[band] >= np.percentile(outer[band], 60)]
        out.append(np.median(pts[band], 0))
    return np.array(out)


def detect(v, kind, sk, parts="auto", weights=None, names=None, segments: int = 3, faces=None):
    """[(Chain, vertex mask, per-vertex looseness)] of the dangling parts."""
    want = {"cape", "hair", "tail"} if parts == "auto" else set(parts)
    unknown = want - {"cape", "hair", "tail"}
    if unknown:
        raise ValueError(f"Unknown spring part(s): {', '.join(sorted(unknown))} (cape, hair, tail)")
    H = sk.height
    loose, d, r = looseness(v, kind, sk)
    j = sk.joints
    found = []
    min_count = max(30, int(0.004 * len(v)))
    if kind == "humanoid":
        torso = (v[:, 1] > j["hips"][1]) & (v[:, 1] < j["neck"][1]) & (np.abs(v[:, 0] - j["chest"][0]) < 0.12 * H)
        front_z = float(np.percentile(v[torso, 2], 95)) if torso.any() else float(j["chest"][2])
        if "cape" in want:
            if faces is not None:                         # thin sheets (side flaps) count too
                sheet = 1 - _smoothstep((sheet_thickness(v, faces, H) - 0.02) / 0.03)
                loose = np.maximum(loose, sheet)
            legs = [k for k, c in enumerate(_core(kind, sk)) if c[0][:5] in ("thigh", "shin_")]
            off_legs = (d[:, legs] - r[legs]).min(1) / H          # tunic panels stay on the legs
            amount = (loose * _smoothstep((front_z - 0.04 * H - v[:, 2]) / (0.04 * H))
                      * _smoothstep((j["neck"][1] - v[:, 1]) / (0.04 * H))
                      * _smoothstep((off_legs - 0.02) / 0.05))
            amount = np.clip(smooth_field(v, amount, H), 0, 1)
            if (amount > 0.5).sum() >= min_count:
                found += _cape_chains(v, amount, sk, segments)
        if "hair" in want and not (parts == "auto" and found):   # a cape's collar isn't hair
            hz = float(j["head"][2])
            amount = (loose * _smoothstep((v[:, 1] - (j["neck"][1] - 0.12 * H)) / (0.04 * H))
                      * _smoothstep((hz - v[:, 2]) / (0.03 * H))
                      * _smoothstep((0.2 * H - np.abs(v[:, 0] - j["head"][0])) / (0.04 * H)))
            amount = np.clip(smooth_field(v, amount, H, sigma=0.02), 0, 1)
            m = amount > 0.5
            if m.sum() >= min_count:
                pts = v[m]
                top = pts[np.argmax(pts[:, 1])]
                anchor = np.array([j["head"][0], min(top[1], j["head"][1] + 0.05 * H),
                                   float(np.percentile(pts[:, 2], 80))])
                key = anchor[1] - pts[:, 1] + 0.5 * np.abs(pts[:, 2] - anchor[2])
                ch = Chain("hair", "head", _chain_along(pts, anchor, segments, key))
                found.append((ch, amount > 0.02, amount))
    else:
        if "tail" in want and getattr(sk, "has_tail", False) and weights is not None:
            idx, w = weights
            tail = [names.index(n) for n in ("tail1", "tail2") if n in names]
            share = np.where(np.isin(idx, tail), w, 0).sum(1)
            m = share > 0.3
            if m.sum() >= min_count // 2:
                pts = v[m]
                anchor = j["tail1"].copy()
                dist = np.linalg.norm(pts - anchor, axis=1)
                ch = Chain("tail", "tail1", _chain_along(pts, anchor, segments, dist))
                found.append((ch, share > 0.02, np.clip(share / 0.6, 0, 1)))
    return found


def _cape_chains(v, amount, sk, n):
    """One chain down the back, or left / back / right when the cape wraps round the sides."""
    j = sk.joints
    H = sk.height
    c = np.array([j["chest"][0], 0.0, j["chest"][2]])
    strong = np.nonzero(amount > 0.5)[0]
    pts = v[strong]
    ang = np.degrees(np.arctan2(pts[:, 0] - c[0], -(pts[:, 2] - c[2])))   # 0 = behind, + = left
    lo, hi = np.percentile(ang, [5, 95])
    if hi - lo > 110:
        centres = np.percentile(ang, [12, 50, 88])
        labels = ("cape_r", "cape", "cape_l")
    else:
        centres = np.array([np.median(ang)])
        labels = ("cape",)
    out = []
    near = np.argmin(np.abs(ang[:, None] - centres[None, :]), 1)
    skin = amount > 0.02                     # every chain of the cape sees every cape vertex
    for k, name in enumerate(labels):
        sp = pts[near == k]
        if len(sp) < 10:
            continue
        top_y = float(np.percentile(sp[:, 1], 97))
        top = sp[sp[:, 1] > top_y - 0.03 * H]
        anchor = np.median(top, 0)
        anchor[1] = min(top_y, float(j["neck"][1]))
        key = anchor[1] - sp[:, 1]
        radial = np.hypot(sp[:, 0] - c[0], sp[:, 2] - c[2]) if len(labels) > 1 else None
        out.append((Chain(name, "chest", _chain_along(sp, anchor, n, key, radial)), skin, amount))
    return out


# --------------------------------------------------------------------------- skinning

def _chain_param(v, pts):
    """Position along a polyline in segment units (0 = anchor, n = tip) and distance to it."""
    n = len(pts) - 1
    best_d = np.full(len(v), np.inf)
    best_t = np.zeros(len(v))
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        u = np.clip(((v - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        dd = np.linalg.norm(v - (a + u[:, None] * ab), axis=1)
        better = dd < best_d
        best_d[better] = dd[better]
        best_t[better] = i + u[better]
    return best_t, best_d


def spring_weights(v, idx, w, found, joint_index: dict, influences: int = 8):
    """New (joint indices, weights): chain bones blended into the old weights."""
    J = max(joint_index.values()) + 1
    dense = np.zeros((len(v), J), np.float32)
    np.add.at(dense, (np.repeat(np.arange(len(v)), idx.shape[1]), idx.reshape(-1)), w.reshape(-1))
    total_chain = np.zeros((len(v), J), np.float32)
    share = np.zeros(len(v))
    # chains of one part (cape_l / cape / cape_r) share their vertices by closeness
    groups: dict[str, list] = {}
    for ch, mask, loose in found:
        groups.setdefault(ch.name.split("_")[0], []).append((ch, mask, loose))
    for _, members in groups.items():
        mask = np.any([m for _, m, _ in members], 0)
        sel = np.nonzero(mask)[0]
        if not len(sel):
            continue
        vs = v[sel]
        params, dists = zip(*[_chain_param(vs, ch.points) for ch, _, _ in members])
        dists = np.stack(dists, 1)
        spacing = np.median([np.linalg.norm(a.points[-1] - a.points[0]) for a, _, _ in members]) * 0.25
        cw = np.exp(-(dists / max(spacing, 1e-6)) ** 2)
        cw /= np.maximum(cw.sum(1, keepdims=True), 1e-12)
        loose = members[0][2][sel]
        tmin = np.min(np.stack(params, 1), 1)
        c = loose * _smoothstep(tmin / 0.7)                         # body near the seam
        for k, (ch, _, _) in enumerate(members):
            t = params[k]
            n = len(ch.points) - 1
            for i, name in enumerate(ch.joints):
                hat = np.clip(1 - np.abs(t - (i + 0.5)), 0, 1)
                if i == 0:
                    hat = np.where(t < 0.5, 1.0, hat)
                if i == n - 1:
                    hat = np.where(t > n - 0.5, 1.0, hat)
                total_chain[sel, joint_index[name]] += (c * cw[:, k] * hat).astype(np.float32)
        share[sel] = np.maximum(share[sel], c)
    s = total_chain.sum(1)
    has = s > 1e-6
    total_chain[has] /= s[has, None]
    blend = dense * (1 - share[:, None]) + total_chain * share[:, None]
    order = np.argsort(-blend, 1)[:, :influences]
    top = np.take_along_axis(blend, order, 1)
    top /= np.maximum(top.sum(1, keepdims=True), 1e-12)
    return order.astype(np.uint8), top.astype(np.float32)


# --------------------------------------------------------------------------- simulation

@dataclass
class Params:
    stiffness: float = 0.5
    damping: float = 0.3
    gravity: float = 1.0
    wind: float = 1.0
    collide: bool = True

    def k(self):
        return 12.0 + 140.0 * self.stiffness ** 1.5      # 1/s^2

    def c(self):
        return 2 * math.sqrt(self.k()) * (0.08 + 0.92 * self.damping)


def _skin_mats(rig, clip, t):
    """World matrices and skinning matrices (world @ inverse bind) by node index at time t."""
    w = rig.worlds(clip, t)
    joints, ibm = rig.skins[0]
    s = np.tile(np.eye(4), (len(w), 1, 1))
    s[joints] = w[joints] @ ibm
    return w, s


def _apply(m, p):
    return p @ m[:3, :3].T + m[:3, 3]


def simulate(rig, clip, chains: list[Chain], caps, radii, H: float, params: Params,
             speed: float = 0.0, times=None, loop: bool | None = None, rate: float = 240.0):
    """{chain name: (keys, n + 1, 3)} world positions of each chain at the key times."""
    names = rig.names
    node = {n: i for i, n in enumerate(names)}
    loop = clip.loop if loop is None else loop
    times = np.asarray(times if times is not None else np.linspace(clip.start, clip.end, 25))
    dur = max(clip.duration, 1e-6)
    k, c = params.k(), params.c()
    g = 9.81 * H / 1.8 * params.gravity
    down = np.array([0, -1.0, 0])
    air = np.array([0, 0, -1.0]) * speed * params.wind * 5.0
    # rest distances from the capsules (the cape may start closer than a capsule's radius)
    caps_pts = [(node.get(owner), a, b) for _, owner, a, b in caps]
    states = []
    for ch in chains:
        n = len(ch.points) - 1
        L = np.linalg.norm(np.diff(ch.points, axis=0), axis=1)
        allow = np.array([[min(r * 1.05, 0.92 * _seg_dist(p[None], a, b)[0]) for (_, a, b), r
                           in zip(caps_pts, radii)] for p in ch.points])
        ki = k * (1.0 - 0.45 * np.arange(n + 1) / max(n, 1))
        fall = np.arange(n + 1) / max(n, 1)
        fall = fall * PART_WIND.get(ch.name.split("_")[0], 1.0)
        states.append(dict(ch=ch, L=L, allow=allow, ki=ki, fall=fall, p=node[ch.parent]))

    def pose_at(t):
        _, s = _skin_mats(rig, clip, t)
        return s

    def targets(s, st):
        m = s[st["p"]]
        return _apply(m, st["ch"].points), m[:3, :3]

    def capsules(s):
        out = []
        for (owner, a, b) in caps_pts:
            m = s[owner] if owner is not None else np.eye(4)
            out.append((_apply(m, a[None])[0], _apply(m, b[None])[0]))
        return out

    dt = 1.0 / rate
    if loop:
        t_run = np.arange(0, 3 * dur + 1e-9, dt)
        settle = 0.0
    else:
        settle = 1.2
        t_run = np.arange(-settle, dur + 1e-9, dt)

    def clip_t(t):
        if loop:
            return clip.start + (t % dur)
        return clip.start + min(max(t, 0.0), dur)
    s0 = pose_at(clip_t(t_run[0]))
    x = [targets(s0, st)[0].copy() for st in states]
    xp = [xi.copy() for xi in x]
    prev_tgt = [targets(s0, st)[0] for st in states]
    rec_t, rec = [], [[] for _ in states]
    record_from = 2 * dur if loop else 0.0
    for t in t_run:
        s = pose_at(clip_t(t))
        cap_now = capsules(s) if params.collide else []
        for si, st in enumerate(states):
            tgt, rot = targets(s, st)
            vel_t = (tgt - prev_tgt[si]) / dt
            prev_tgt[si] = tgt
            v = (x[si] - xp[si]) / dt
            grav = g * (down - rot @ down)
            damp = c * (1.6 if t < 0 else 1.0)
            acc = st["ki"][:, None] * (tgt - x[si]) - damp * (v - vel_t) + grav[None, :] \
                + (air[None, :] * st["fall"][:, None] if t >= 0 else air[None, :] * 0)
            new = x[si] + v * dt + acc * dt * dt
            new[0] = tgt[0]                                        # the anchor rides the body
            for _ in range(2):
                for i in range(1, len(new)):                       # bone lengths, swing limit
                    d = new[i] - new[i - 1]
                    dn = d / max(np.linalg.norm(d), 1e-9)
                    r = tgt[i] - tgt[i - 1]
                    rn = r / max(np.linalg.norm(r), 1e-9)
                    cosang = float(dn @ rn)
                    if cosang < COS_MAX:
                        perp = dn - cosang * rn
                        pn = np.linalg.norm(perp)
                        if pn > 1e-9:
                            dn = COS_MAX * rn + SIN_MAX * perp / pn
                    new[i] = new[i - 1] + dn * st["L"][i - 1]
                for ci, (a, b) in enumerate(cap_now):              # out of the body
                    ab = b - a
                    u = np.clip(((new - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
                    close = a + u[:, None] * ab
                    off = new - close
                    dist = np.linalg.norm(off, axis=1)
                    need = st["allow"][:, ci]
                    hit = (dist < need) & (np.arange(len(new)) > 0)
                    if hit.any():
                        dirn = off[hit] / np.maximum(dist[hit, None], 1e-9)
                        new[hit] = close[hit] + dirn * need[hit, None]
                floor = 0.012 * H
                new[1:, 1] = np.maximum(new[1:, 1], floor)
            xp[si], x[si] = x[si], new
        if t >= record_from - 1e-9:
            rec_t.append(t - record_from)
            for si in range(len(states)):
                rec[si].append(x[si].copy())
    rec_t = np.array(rec_t)
    out = {}
    for si, st in enumerate(states):
        r = np.array(rec[si])                                      # (T, n+1, 3)
        rel = times - clip.start
        k0 = np.clip(np.searchsorted(rec_t, rel) - 1, 0, len(rec_t) - 2)
        u = np.clip((rel - rec_t[k0]) / dt, 0, 1)[:, None, None]
        keys = r[k0] * (1 - u) + r[k0 + 1] * u
        if loop:                                                   # blend the seam shut
            s_end = pose_at(clip.end)
            s_start = pose_at(clip.start)
            tgt_e, _ = targets(s_end, st)
            tgt_s, _ = targets(s_start, st)
            # same offset from the body at both ends (the body itself loops)
            err = (keys[0] - tgt_s) - (keys[-1] - tgt_e)
            w = ((times - clip.start) / dur)[:, None, None]
            keys = keys + err[None] * w
            for kk in range(len(keys)):
                for i in range(1, keys.shape[1]):
                    d = keys[kk, i] - keys[kk, i - 1]
                    keys[kk, i] = keys[kk, i - 1] + d * (st["L"][i - 1] / max(np.linalg.norm(d), 1e-9))
        out[st["ch"].name] = keys
    return out


def chain_rotations(chain: Chain, keys_pos: np.ndarray, parent_rot: np.ndarray) -> np.ndarray:
    """Local rotations (keys, n, 4) of the chain bones from simulated positions.
    parent_rot: (keys, 4) world rotation of the chain's parent joint."""
    n = len(chain.points) - 1
    out = np.zeros((len(keys_pos), n, 4))
    for f in range(len(keys_pos)):
        world_prev = parent_rot[f]
        for i in range(n):
            rest_dir = chain.points[i + 1] - chain.points[i]
            now = qrot(world_prev, rest_dir)
            want = keys_pos[f, i + 1] - keys_pos[f, i]
            g = qmul(qarc(now, want), world_prev)
            out[f, i] = qmul(qconj(world_prev), g)
            world_prev = g
    for i in range(n):
        for f in range(1, len(out)):
            if out[f, i] @ out[f - 1, i] < 0:
                out[f, i] = -out[f, i]
    return out


# --------------------------------------------------------------------------- glTF

def _compact(doc: dict, bin_: bytes) -> tuple[dict, bytearray]:
    """Drop accessors and buffer views nothing refers to; repack the binary."""
    used_acc: set[int] = set()
    for mesh in doc.get("meshes", []):
        for p in mesh.get("primitives", []):
            used_acc.update(p.get("attributes", {}).values())
            if "indices" in p:
                used_acc.add(p["indices"])
            for tgt in p.get("targets", []):
                used_acc.update(tgt.values())
    for sk in doc.get("skins", []):
        if "inverseBindMatrices" in sk:
            used_acc.add(sk["inverseBindMatrices"])
    for a in doc.get("animations", []):
        for s in a.get("samplers", []):
            used_acc.update((s["input"], s["output"]))
    acc_map = {old: new for new, old in enumerate(sorted(used_acc))}
    accessors = [doc["accessors"][i] for i in sorted(used_acc)]
    used_view = {a["bufferView"] for a in accessors if "bufferView" in a}
    used_view |= {im["bufferView"] for im in doc.get("images", []) if "bufferView" in im}
    view_map, views, out = {}, [], bytearray()
    for old in sorted(used_view):
        v = dict(doc["bufferViews"][old])
        while len(out) % 4:
            out.append(0)
        data = bin_[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        v["byteOffset"] = len(out)
        out += data
        view_map[old] = len(views)
        views.append(v)
    for a in accessors:
        if "bufferView" in a:
            a["bufferView"] = view_map[a["bufferView"]]
    for im in doc.get("images", []):
        if "bufferView" in im:
            im["bufferView"] = view_map[im["bufferView"]]
    for mesh in doc.get("meshes", []):
        for p in mesh.get("primitives", []):
            p["attributes"] = {k: acc_map[v] for k, v in p["attributes"].items()}
            if "indices" in p:
                p["indices"] = acc_map[p["indices"]]
            if "targets" in p:
                p["targets"] = [{k: acc_map[v] for k, v in t.items()} for t in p["targets"]]
    for sk in doc.get("skins", []):
        if "inverseBindMatrices" in sk:
            sk["inverseBindMatrices"] = acc_map[sk["inverseBindMatrices"]]
    for a in doc.get("animations", []):
        for s in a.get("samplers", []):
            s["input"], s["output"] = acc_map[s["input"]], acc_map[s["output"]]
    doc["accessors"], doc["bufferViews"] = accessors, views
    return doc, out


def _rig_info(doc):
    skins = doc.get("skins", [])
    if not skins:
        return None
    return (skins[0].get("extras") or {}).get("pixel_pipeline_rig")


def has_springs(path) -> bool:
    from .anim3d import Gltf
    info = _rig_info(Gltf(path).doc)
    return bool(info and info.get("springs"))


def _clip_speed(anim: dict, name: str, H: float) -> float:
    extras = anim.get("extras") or {}
    if isinstance(extras.get("speed"), (int, float)):
        return float(extras["speed"])
    base = name.lower().split("_")[0]
    return DEFAULT_SPEED.get(base, 0.0) * H


def add_springs(rigged_glb, out_glb=None, *, parts="auto", stiffness: float = 0.5,
                damping: float = 0.3, gravity: float = 1.0, wind: float = 1.0,
                segments: int = 3, collide: bool = True) -> Path:
    """Spring bones for the dangling parts of an auto-rigged character, baked into every clip.

    parts: "auto" or a list of "cape", "hair", "tail". stiffness 0 (floppy) .. 1 (stiff),
    damping 0 (keeps swinging) .. 1 (settles at once), gravity x real gravity, wind x the air
    of travelling (walk/run clips; mocap clips carry their own speed). A file that already has
    springs is only re-simulated (with these settings). Returns the written path."""
    from .anim3d import Gltf, GlbWriter, Rig
    from .motions import read_skeleton
    src = Path(rigged_glb)
    out = Path(out_glb) if out_glb else src
    params = Params(stiffness, damping, gravity, wind, collide)
    g0 = Gltf(src)
    doc = json.loads(json.dumps(g0.doc))
    if len(doc.get("buffers", [])) != 1 or "uri" in doc["buffers"][0]:
        raise ValueError(f"{src.name}: expected one binary buffer (a .glb made by the auto-rig)")
    bin_ = bytes(g0.buffers[0][:doc["buffers"][0]["byteLength"]])
    kind, sk, _ = read_skeleton(src)
    info = _rig_info(doc)
    rig = Rig(src)
    nodes = doc["nodes"]
    index = {n.get("name"): i for i, n in enumerate(nodes)}
    skin = doc["skins"][0]
    g = GlbWriter()
    g.doc, g.bin = doc, bytearray(bin_)
    if info and info.get("springs"):
        chains = [Chain(c["name"], c["parent"], np.asarray(c["points"], np.float64))
                  for c in info["springs"]["chains"]]
    else:
        prims = rig.primitives
        verts = np.concatenate([p.positions for p in prims])
        skin_nodes = list(skin["joints"])
        names = [nodes[i].get("name") for i in skin_nodes]
        idx_all = np.concatenate([p.joints for p in prims])
        w_all = np.concatenate([p.weights for p in prims])
        offs = np.cumsum([0] + [len(p.positions) for p in prims])
        faces = np.concatenate([p.faces + o for p, o in zip(prims, offs)])
        found = detect(verts, kind, sk, parts, (idx_all, w_all), names, segments, faces)
        if not found:
            raise ValueError("Nothing to swing found (no cape, hair or tail hanging off the body)")
        chains = [f[0] for f in found]
        # new joints: nodes under their parents, appended to the skin
        rest = {n: np.asarray(p, np.float64) for n, p in sk.joints.items()}
        joint_index = {nm: k for k, nm in enumerate(names)}
        ibm_old = rig.skins[0][1]
        ibms = [m for m in ibm_old]
        for ch in chains:
            prev_name, prev_pos = ch.parent, rest[ch.parent]
            for i, jn in enumerate(ch.joints):
                pos = ch.points[i]
                nodes.append({"name": jn, "translation": [float(x) for x in pos - prev_pos]})
                ni = len(nodes) - 1
                nodes[index[prev_name]].setdefault("children", []).append(ni)
                index[jn] = ni
                skin["joints"].append(ni)
                joint_index[jn] = len(skin["joints"]) - 1
                m = np.eye(4)
                m[:3, 3] = -pos
                ibms.append(m)
                rest[jn] = pos
                prev_name, prev_pos = jn, pos
        ibm = np.array(ibms, np.float32)
        skin["inverseBindMatrices"] = g.accessor(ibm.transpose(0, 2, 1).reshape(-1, 16), "MAT4")
        # weights per primitive
        new_idx, new_w = spring_weights(verts, idx_all, w_all, found, joint_index)
        if not (new_w[:, 4:] > 0).any():                 # 4 influences are enough
            new_idx, new_w = new_idx[:, :4], new_w[:, :4]
        start = 0
        mesh_prims = [(nd, p) for nd in nodes if "mesh" in nd and "skin" in nd
                      for p in doc["meshes"][nd["mesh"]]["primitives"]]
        for (nd, p), rp in zip(mesh_prims, prims):
            n = len(rp.positions)
            for k in range(8):
                p["attributes"].pop(f"JOINTS_{k}", None)
                p["attributes"].pop(f"WEIGHTS_{k}", None)
            for k in range(new_idx.shape[1] // 4):          # 8 influences: JOINTS_1 / WEIGHTS_1
                cols = slice(4 * k, 4 * k + 4)
                p["attributes"][f"JOINTS_{k}"] = g.accessor(new_idx[start:start + n, cols], "VEC4", 5121)
                p["attributes"][f"WEIGHTS_{k}"] = g.accessor(new_w[start:start + n, cols], "VEC4")
            start += n
        info = dict(info or {})
        skin.setdefault("extras", {})["pixel_pipeline_rig"] = info
    info["springs"] = {"chains": [{"name": c.name, "parent": c.parent,
                                   "points": np.round(c.points, 6).tolist()} for c in chains],
                       "stiffness": stiffness, "damping": damping, "gravity": gravity, "wind": wind,
                       "collide": collide}
    skin.setdefault("extras", {})["pixel_pipeline_rig"] = info
    # bake every clip
    caps = _core(kind, sk)
    verts0 = np.concatenate([p.positions for p in rig.primitives])
    radii, _ = _radii(verts0, caps)
    spring_nodes = {index[jn] for ch in chains for jn in ch.joints}
    by_name = {c.name: c for c in rig.clips}
    jobs = []
    for a in doc.get("animations", []):
        # drop old spring channels (rebake)
        keep_ch = [ch for ch in a["channels"] if ch["target"].get("node") not in spring_nodes]
        used = sorted({ch["sampler"] for ch in keep_ch})
        remap = {o: n for n, o in enumerate(used)}
        a["samplers"] = [a["samplers"][o] for o in used]
        for ch in keep_ch:
            ch["sampler"] = remap[ch["sampler"]]
        a["channels"] = keep_ch
        clip = by_name.get(_clip_name(a, src, len(doc["animations"])))
        if clip is None or not a["samplers"]:
            continue
        t_acc = a["samplers"][0]["input"]
        times = g0.accessor(t_acc).reshape(-1).astype(np.float64) if t_acc < len(g0.doc["accessors"]) \
            else np.linspace(clip.start, clip.end, 25)
        speed = _clip_speed(a, clip.name, sk.height)
        jobs.append((a, clip, t_acc, times, speed))
    spring_names = {jn for ch in chains for jn in ch.joints}
    all_keys = simulate_clips(src, rig, [(clip, times, speed) for _, clip, _, times, speed in jobs],
                              chains, caps, radii, sk.height, params, spring_names)
    for (a, clip, t_acc, times, speed), keys in zip(jobs, all_keys):
        for ch in chains:
            prot = np.array([qfrom_matrix(rig.worlds(clip, t)[index[ch.parent]][:3, :3]) for t in times])
            rots = chain_rotations(ch, keys[ch.name], prot)
            for i, jn in enumerate(ch.joints):
                a["samplers"].append({"input": t_acc,
                                      "output": g.accessor(rots[:, i].astype(np.float32), "VEC4")})
                a["channels"].append({"sampler": len(a["samplers"]) - 1,
                                      "target": {"node": index[jn], "path": "rotation"}})
    doc, packed = _compact(doc, bytes(g.bin))
    g.doc, g.bin = doc, packed
    tmp = out.with_name(out.name + ".tmp")
    g.save(tmp)
    tmp.replace(out)
    return out


_SIM_RIG: dict = {}


def _sim_init(path: str) -> None:
    from .anim3d import Rig
    _SIM_RIG["rig"] = Rig(path)


def _sim_task(job):
    clip_name_, chains, caps, radii, H, params, speed, times = job
    rig = _SIM_RIG["rig"]
    clip = next(c for c in rig.clips if c.name == clip_name_)
    return simulate(rig, clip, chains, caps, radii, H, params, speed, times)


def _sim_key(rig, clip, chains, caps, radii, H, params, speed, times, spring_names) -> str:
    """Everything simulate() reads for a clip: the rig's rest pose, hierarchy and skin, the
    clip's channels (the spring joints' own left out: an earlier bake's swing never moves the
    body they hang from), the chains, capsules and settings, and the code."""
    import dataclasses

    from . import cache
    skip = {i for i, nm in enumerate(rig.names) if nm in spring_names}
    body = [i for i in range(len(rig.names)) if i not in skip]       # (the springs' own
    chans = [[c.node, c.path, c.times, c.values, c.interpolation] for c in clip.channels
             if c.node not in skip]                                  # joints never matter)
    code = cache.source_of(simulate, _skin_mats, _apply, _seg_dist)
    skin = None
    if rig.skins:
        joints, ibm = rig.skins[0]
        keep = [k for k, j in enumerate(joints) if int(j) not in skip]
        skin = [np.asarray(joints)[keep], np.asarray(ibm)[keep]]
    return cache.digest([[rig.names[i]] + list(rig.rest[i]) for i in body],
                        [int(rig.parent[i]) for i in body], [i for i in rig.order if i not in skip],
                        skin, chans, clip.start, clip.end,
                        clip.loop, [[c.name, c.parent, c.points] for c in chains],
                        [list(c) for c in caps], radii, H, dataclasses.astuple(params), speed,
                        times, code, COS_MAX, SIN_MAX, PART_WIND, cache.code_digest("anim3d"))


def simulate_clips(src, rig, clips, chains, caps, radii, H: float, params: Params,
                   spring_names=()) -> list[dict]:
    """simulate() for each (clip, times, speed): clips simulated before with the same
    everything come from the cache (cache.py; a rebake after adding clips simulates only the
    new ones), the rest on worker processes when there are several (the same numbers)."""
    from . import cache, parallel
    out: list = [None] * len(clips)
    keys: list = [None] * len(clips)
    if cache.enabled():
        for i, (clip, times, speed) in enumerate(clips):
            keys[i] = _sim_key(rig, clip, chains, caps, radii, H, params, speed, times,
                               set(spring_names))
            hit = cache.load("springs", keys[i])
            if hit is not None and all(f"c{j}" in hit for j in range(len(chains))):
                out[i] = {ch.name: hit[f"c{j}"] for j, ch in enumerate(chains)}
    todo = [i for i in range(len(clips)) if out[i] is None]
    jobs = [(clips[i][0].name, chains, caps, radii, H, params, clips[i][2], clips[i][1])
            for i in todo]
    workers = parallel.worker_count(len(jobs), per_worker_mb=300) if len(jobs) >= 3 else 1
    if workers > 1:
        res = parallel.run(_sim_task, jobs, init=_sim_init, init_args=(str(src),),
                           workers=workers)
    else:
        res = [simulate(rig, clips[i][0], chains, caps, radii, H, params, clips[i][2],
                        clips[i][1]) for i in todo]
    for i, r in zip(todo, res):
        out[i] = r
        if keys[i] is not None:
            cache.save("springs", keys[i], **{f"c{j}": r[ch.name] for j, ch in enumerate(chains)})
    return out


def _clip_name(anim: dict, src: Path, count: int) -> str:
    from .anim3d import clip_name
    return clip_name(anim.get("name"), src, count)


def rebake(rigged_glb, out_glb=None) -> Path:
    """Re-simulate the springs of a file that has them (e.g. after clips were added), with the
    settings it was made with."""
    from .anim3d import Gltf
    info = _rig_info(Gltf(rigged_glb).doc) or {}
    sp = info.get("springs")
    if not sp:
        raise ValueError(f"{Path(rigged_glb).name} has no springs")
    return add_springs(rigged_glb, out_glb, stiffness=sp.get("stiffness", 0.5),
                       damping=sp.get("damping", 0.3), gravity=sp.get("gravity", 1.0),
                       wind=sp.get("wind", 1.0), collide=sp.get("collide", True))

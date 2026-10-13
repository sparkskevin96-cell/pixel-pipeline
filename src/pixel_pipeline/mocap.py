"""Motion capture (BVH) retargeted onto the auto-rigged humanoid (autorig.py, 17 joints).

    retarget_bvh("02_01.bvh", "rigged.glb", "walk")      # a real walk, appended as a clip
    library.add_mocap(character, ["walk", "run", "sword slash"])   # by friendly name

1. BVH: hierarchy (OFFSET, CHANNELS in any Euler order, End Site), MOTION (frames, frame time).
   Forward kinematics gives every BVH joint's world rotation and position per frame.
2. Names: CMU (Hips, LHipJoint, LeftUpLeg, LeftLeg, LeftFoot, LowerBack, Spine, Spine1, Neck,
   Neck1, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand...), Mixamo (mixamorig:Hips,
   LeftUpLeg, Spine2...), 3ds Max biped (Bip01 L Thigh...) and plain names are matched to our
   17 joints. Joints in between (LHipJoint, LowerBack, Neck1, shoulders) need no match: world
   rotations already include them.
3. Rest poses differ (BVH T-pose, our rig stands world-aligned with the arms however the model
   held them): each of our limb bones first turns from its own rest direction onto the BVH
   bone's rest direction (a per-bone rest offset quaternion), then takes the BVH bone's world
   rotation since its rest. The BVH's axes are found from its rest pose (up = hips to head,
   left = right hip to left hip) and turned into ours (y up, front +z, left +x), and the clip is
   turned about y so the character faces +z on average.
4. Root motion is scaled by leg length (ours / the actor's) and grounded (the lowest foot
   reaches our rest foot height); in_place removes the steady travel (kept as extras.speed,
   which springs.py uses as the air rushing past). Optional foot planting: while a foot is down
   and slow it is pinned to one spot and the leg reaches it by 2-bone IK.
5. Looping clips: the best cycle in the range (pose + velocity match) is cut out and the small
   remaining mismatch is spread over the cycle, so the last key equals the first.
6. Keys are resampled to our clip format (25 keys by default) and appended with the same
   writer as motions.add_animations (old clips stay byte for byte).

Only the humanoid rig takes mocap (no four-legged BVH source is wired up).
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- quaternions (x y z w, vectorised)


def qmul(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qconj(q):
    q = np.asarray(q, np.float64)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrot(q, v):
    q, v = np.asarray(q, np.float64), np.asarray(v, np.float64)
    u, w = q[..., :3], q[..., 3:4]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qaxis(axis, rad):
    """Quaternions about a unit axis (3,) by angles rad (...)."""
    rad = np.asarray(rad, np.float64)
    h = rad / 2
    return np.concatenate([np.multiply.outer(np.sin(h), axis), np.cos(h)[..., None]], -1)


def qarc(a, b):
    """Shortest rotations turning directions a onto b (broadcast over leading axes)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    d = np.sum(a * b, -1, keepdims=True)
    c = np.cross(a, b)
    q = np.concatenate([c, 1.0 + d], -1)
    flip = d[..., 0] < -0.9999                   # opposite: any perpendicular axis, 180 degrees
    if np.any(flip):
        af = a[flip]
        other = np.where(np.abs(af[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]])
        ax = np.cross(af, other)
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        q[flip] = np.concatenate([ax, np.zeros((len(ax), 1))], 1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qnorm(q):
    q = np.asarray(q, np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def qslerp(a, b, u):
    """Slerp a -> b by u (arrays broadcast; u (...))."""
    a, b = qnorm(a), qnorm(b)
    u = np.asarray(u, np.float64)[..., None]
    d = np.sum(a * b, -1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-6
    wa = np.where(small, 1 - u, np.sin((1 - u) * th) / np.where(small, 1, s))
    wb = np.where(small, u, np.sin(u * th) / np.where(small, 1, s))
    return qnorm(wa * a + wb * b)


def qfrom_matrix(m):
    from .anim3d import _matrix_quat
    return qnorm(_matrix_quat(np.asarray(m, np.float64)))


def _hemisphere(q):
    """Keep consecutive quaternions (F, 4) on one hemisphere."""
    q = np.array(q, np.float64)
    for k in range(1, len(q)):
        if q[k] @ q[k - 1] < 0:
            q[k] = -q[k]
    return q


# --------------------------------------------------------------------------- BVH


@dataclass
class BvhJoint:
    name: str
    parent: int                       # -1 for the root
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    end: np.ndarray | None = None     # End Site offset
    children: list[int] = field(default_factory=list)


@dataclass
class Bvh:
    joints: list[BvhJoint]
    frame_time: float
    frames: np.ndarray                # (frames, channels)
    path: Path | None = None

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def duration(self) -> float:
        return max(0, len(self.frames) - 1) * self.frame_time

    def index(self, name: str) -> int:
        return self.names.index(name)

    def local(self) -> tuple[np.ndarray, np.ndarray]:
        """(rotations (F, J, 4), translations (F, J, 3)) of every joint relative to its parent."""
        F, J = len(self.frames), len(self.joints)
        rot = np.zeros((F, J, 4))
        rot[..., 3] = 1
        tr = np.zeros((F, J, 3))
        col = 0
        axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
        for j, jt in enumerate(self.joints):
            tr[:, j] = jt.offset
            for ch in jt.channels:
                v = self.frames[:, col]
                col += 1
                kind, axis = ch[1:].lower(), ch[0].lower()
                if kind == "rotation":            # channel order = multiplication order
                    rot[:, j] = qmul(rot[:, j], qaxis(axes[axis], np.radians(v)))
                elif kind == "position":
                    tr[:, j, "xyz".index(axis)] += v
        return rot, tr

    def world(self) -> tuple[np.ndarray, np.ndarray]:
        """(world rotations (F, J, 4), world positions (F, J, 3))."""
        lr, lt = self.local()
        g = np.zeros_like(lr)
        p = np.zeros_like(lt)
        for j, jt in enumerate(self.joints):
            if jt.parent < 0:
                g[:, j], p[:, j] = lr[:, j], lt[:, j]
            else:
                g[:, j] = qmul(g[:, jt.parent], lr[:, j])
                p[:, j] = p[:, jt.parent] + qrot(g[:, jt.parent], lt[:, j])
        return g, p

    def rest(self) -> np.ndarray:
        """(J, 3) joint positions with every channel at zero (offsets only)."""
        p = np.zeros((len(self.joints), 3))
        for j, jt in enumerate(self.joints):
            p[j] = jt.offset + (p[jt.parent] if jt.parent >= 0 else 0)
        return p

    def rest_end(self, j: int) -> np.ndarray | None:
        jt = self.joints[j]
        return None if jt.end is None else self.rest()[j] + jt.end


def parse_bvh(source) -> Bvh:
    """A BVH file (path) or its text."""
    path = None
    if isinstance(source, (str, Path)) and "\n" not in str(source) and Path(source).exists():
        path = Path(source)
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = str(source)
    tok = text.split()
    i = 0
    joints: list[BvhJoint] = []
    stack: list[int] = []
    pending_end = False

    def nums(k):
        nonlocal i
        out = np.array([float(x) for x in tok[i:i + k]])
        i += k
        return out
    if not tok or tok[0].upper() != "HIERARCHY":
        raise ValueError("Not a BVH file (no HIERARCHY)")
    i = 1
    while i < len(tok) and tok[i].upper() != "MOTION":
        t = tok[i]
        up = t.upper()
        if up in ("ROOT", "JOINT"):
            name = tok[i + 1]
            i += 2
            # names with spaces ("Bip01 L Thigh"): read up to the brace
            while tok[i] != "{":
                name += " " + tok[i]
                i += 1
            parent = stack[-1] if stack else -1
            joints.append(BvhJoint(name, parent, np.zeros(3)))
            if parent >= 0:
                joints[parent].children.append(len(joints) - 1)
        elif up == "END":
            pending_end = True
            i += 2
            while tok[i] != "{":
                i += 1
        elif t == "{":
            if pending_end:
                stack.append(-2)              # End Site block
            else:
                stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            pending_end = False
            i += 1
        elif up == "OFFSET":
            i += 1
            off = nums(3)
            if stack and stack[-1] == -2:
                owner = next(s for s in reversed(stack) if s >= 0)
                joints[owner].end = off
            else:
                joints[-1].offset = off
        elif up == "CHANNELS":
            n = int(tok[i + 1])
            joints[-1].channels = tok[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    if not joints:
        raise ValueError("BVH has no joints")
    # MOTION
    i += 1
    count = None
    frame_time = 1 / 30
    while i < len(tok):
        t = tok[i].rstrip(":").lower()
        if t == "frames":
            count = int(tok[i + 1])
            i += 2
        elif t == "frame" and i + 1 < len(tok) and tok[i + 1].rstrip(":").lower() == "time":
            frame_time = float(tok[i + 2])
            i += 3
            break
        else:
            i += 1
    nch = sum(len(j.channels) for j in joints)
    data = np.array([float(x) for x in tok[i:]], np.float64)
    rows = len(data) // max(nch, 1)
    if count is not None:
        rows = min(rows, count)
    frames = data[:rows * nch].reshape(rows, nch) if nch else np.zeros((rows, 0))
    if rows == 0:
        raise ValueError("BVH has no frames")
    return Bvh(joints, frame_time if frame_time > 0 else 1 / 30, frames, path)


def write_bvh(path, joints: list[BvhJoint], frames: np.ndarray, frame_time: float) -> Path:
    """Write a BVH (tests and tools): joints parents first."""
    lines = ["HIERARCHY"]

    def emit(j, depth):
        jt = joints[j]
        pad = "  " * depth
        lines.append(f"{pad}{'ROOT' if jt.parent < 0 else 'JOINT'} {jt.name}")
        lines.append(pad + "{")
        lines.append(f"{pad}  OFFSET {' '.join(f'{x:.6f}' for x in jt.offset)}")
        lines.append(f"{pad}  CHANNELS {len(jt.channels)} {' '.join(jt.channels)}")
        for c in jt.children:
            emit(c, depth + 1)
        if not jt.children or jt.end is not None:
            e = jt.end if jt.end is not None else np.zeros(3)
            lines.extend([f"{pad}  End Site", pad + "  {", f"{pad}    OFFSET {' '.join(f'{x:.6f}' for x in e)}",
                          pad + "  }"])
        lines.append(pad + "}")
    emit(0, 0)
    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {frame_time:.7f}"]
    lines += [" ".join(f"{x:.5f}" for x in row) for row in frames]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


# --------------------------------------------------------------------------- names

# our joint -> BVH names (compared lower case, without prefixes like "mixamorig:" / "Bip01 ")
NAME_MAP = {
    "hips": ["hips", "hip", "pelvis", "root", "bip01"],
    "spine": ["spine", "lowerback", "spine0", "abdomen", "spine01"],
    "chest": ["spine1", "spine2", "chest", "upperchest", "spine02", "thorax"],
    "neck": ["neck", "neck1", "neck01"],
    "head": ["head"],
    "upper_arm_l": ["leftarm", "leftupperarm", "lupperarm", "lshldr", "larm", "upperarm_l", "l upperarm",
                    "left_shoulder_joint", "leftshoulderarm"],
    "forearm_l": ["leftforearm", "leftlowerarm", "lforearm", "lforearm", "lowerarm_l", "l forearm",
                  "leftelbow"],
    "hand_l": ["lefthand", "lhand", "hand_l", "l hand", "leftwrist"],
    "upper_arm_r": ["rightarm", "rightupperarm", "rupperarm", "rshldr", "rarm", "upperarm_r", "r upperarm",
                    "right_shoulder_joint", "rightshoulderarm"],
    "forearm_r": ["rightforearm", "rightlowerarm", "rforearm", "lowerarm_r", "r forearm", "rightelbow"],
    "hand_r": ["righthand", "rhand", "hand_r", "r hand", "rightwrist"],
    "thigh_l": ["leftupleg", "leftthigh", "lthigh", "leftupperleg", "thigh_l", "l thigh", "lefthip"],
    "shin_l": ["leftleg", "leftshin", "lshin", "leftlowerleg", "calf_l", "l calf", "leftknee"],
    "foot_l": ["leftfoot", "lfoot", "foot_l", "l foot", "leftankle"],
    "thigh_r": ["rightupleg", "rightthigh", "rthigh", "rightupperleg", "thigh_r", "r thigh", "righthip"],
    "shin_r": ["rightleg", "rightshin", "rshin", "rightlowerleg", "calf_r", "r calf", "rightknee"],
    "foot_r": ["rightfoot", "rfoot", "foot_r", "r foot", "rightankle"],
}
TOE = {"l": ["lefttoebase", "lefttoe", "ltoe", "toe_l", "l toe0", "lefttoe_end"],
       "r": ["righttoebase", "righttoe", "rtoe", "toe_r", "r toe0", "righttoe_end"]}
LIMBS = ("upper_arm_l", "forearm_l", "hand_l", "upper_arm_r", "forearm_r", "hand_r",
         "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r")


def _clean(name: str) -> str:
    n = name.lower()
    n = n.split(":")[-1].split("|")[-1]
    n = re.sub(r"^(bip0?1|bip|mixamorig|biped|character\d*|ch\d+)[ _]*", "", n)
    return n.strip()


def match_joints(bvh: Bvh) -> dict[str, int]:
    """Our joint name -> BVH joint index (only those found)."""
    cleaned = [_clean(n) for n in bvh.names]
    out: dict[str, int] = {}
    for ours, cands in NAME_MAP.items():
        for c in cands:
            hits = [k for k, n in enumerate(cleaned) if n == c or n.replace("_", "") == c.replace("_", "")]
            if hits:
                out[ours] = hits[0]
                break
    if "hips" not in out:
        out["hips"] = next(k for k, j in enumerate(bvh.joints) if j.parent < 0)
    # CMU: "Spine" sits above "LowerBack"; prefer Spine for spine when both exist
    for k, n in enumerate(cleaned):
        if n == "spine" and out.get("spine") != k:
            out["spine"] = k
    if out.get("chest") == out.get("spine"):
        out.pop("chest")
    missing = [n for n in ("thigh_l", "shin_l", "thigh_r", "shin_r") if n not in out]
    if missing:
        raise ValueError(f"BVH skeleton not recognised: no {', '.join(missing)} "
                         f"(joints: {', '.join(bvh.names[:30])})")
    return out


def _toe(bvh: Bvh, side: str, foot: int) -> np.ndarray | None:
    cleaned = [_clean(n) for n in bvh.names]
    rest = bvh.rest()
    for c in TOE[side]:
        if c in cleaned:
            return rest[cleaned.index(c)]
    kids = bvh.joints[foot].children
    if kids:
        return rest[kids[0]]
    return bvh.rest_end(foot)


# --------------------------------------------------------------------------- retargeting

@dataclass
class Retargeted:
    """Retargeted motion at the BVH's frame rate (before cutting to keys)."""
    local: dict[str, np.ndarray]      # our joint -> (F, 4) local rotation
    offset: np.ndarray                # (F, 3) hips offset from its rest position (model units)
    fps: float
    speed: float = 0.0                # steady travel removed by in_place (units / s)
    notes: list[str] = field(default_factory=list)


def _our_fk(names, parent, rest, local, offset):
    """World rotations / positions (F, J) of our joints."""
    F = len(offset)
    g, p = {}, {}
    for n in names:
        par = parent[n]
        if par is None:
            g[n] = local[n]
            p[n] = rest[n] + offset
        else:
            g[n] = qmul(g[par], local[n])
            p[n] = p[par] + qrot(g[par], np.broadcast_to(rest[n] - rest[par], (F, 3)))
    return g, p


def retarget(bvh: Bvh, sk, *, start: float | None = None, end: float | None = None,
             in_place: bool = True, loop: bool | None = None, plant_feet: bool = True,
             max_shot: float = 2.5, cycle: tuple[float, float] | None = None) -> tuple[Retargeted, bool]:
    """BVH -> our humanoid skeleton (autorig.Skeleton). Returns (motion, loops).
    A one-shot longer than max_shot seconds with no end given is cut to its main move
    (action_window): an 18 s swordplay take squeezed into 8 frames is noise.
    cycle: (shortest, longest) loop in seconds (default 0.45-3)."""
    from .autorig import NAMES, PARENT
    m = match_joints(bvh)
    notes: list[str] = []
    gb, pb = bvh.world()
    rest_b = bvh.rest()
    F = len(gb)
    # ---- frame range
    f0 = 0 if start is None else int(round(np.clip(start, 0, bvh.duration) * bvh.fps))
    f1 = F - 1 if end is None else int(round(np.clip(end, 0, bvh.duration) * bvh.fps))
    if f1 <= f0:
        raise ValueError(f"Empty range {start}..{end} (the clip is {bvh.duration:.2f} s)")
    # ---- the BVH's axes -> ours (left +x, up +y, front +z)
    hb = rest_b[m["hips"]]
    top = rest_b[m["head"]] if "head" in m else rest_b[m.get("neck", m.get("chest", m["spine"]))]
    up = top - hb
    up /= np.linalg.norm(up)
    left = rest_b[m["thigh_l"]] - rest_b[m["thigh_r"]]
    left -= up * (left @ up)
    if np.linalg.norm(left) < 1e-9:
        raise ValueError("BVH rest pose: the hips have no width")
    left /= np.linalg.norm(left)
    fwd = np.cross(left, up)
    basis = np.stack([left, up, fwd])                  # rows: our axes in BVH coordinates
    mq = qfrom_matrix(basis)                           # turns BVH vectors into ours
    mqc = qconj(mq)

    def conv_q(q):                                    # a BVH world rotation in our axes
        return qmul(qmul(mq, q), mqc)
    rest_o = (basis @ rest_b.T).T                     # BVH rest positions in our axes
    pos_o = np.einsum("ij,fkj->fki", basis, pb)

    # ---- scale by leg length
    def leg(rp, a, b, c):
        return np.linalg.norm(rp[a] - rp[b]) + np.linalg.norm(rp[b] - rp[c])
    ours = np.mean([np.linalg.norm(sk.joints[f"thigh_{s}"] - sk.joints[f"shin_{s}"])
                    + np.linalg.norm(sk.joints[f"shin_{s}"] - sk.joints[f"foot_{s}"]) for s in "lr"])
    theirs = np.mean([leg(rest_o, m[f"thigh_{s}"], m[f"shin_{s}"], m.get(f"foot_{s}", m[f"shin_{s}"]))
                      for s in "lr"])
    scale = float(ours / max(theirs, 1e-9))

    # ---- rest offsets: our limb bone onto the BVH bone's rest direction
    child = {"upper_arm": "forearm", "forearm": "hand", "thigh": "shin", "shin": "foot"}
    align = {}
    for n in NAMES:
        align[n] = np.array([0.0, 0, 0, 1])
        if n not in LIMBS or n not in m:
            continue
        part, s = n[:-2], n[-1]
        if part in child:
            c = f"{child[part]}_{s}"
            if c not in m:
                continue
            ours_d = sk.joints[c] - sk.joints[n]
            theirs_d = rest_o[m[c]] - rest_o[m[n]]
        elif part == "hand":
            tip = sk.tips.get(f"finger_{s}")
            kids = bvh.joints[m[n]].children
            end_b = (rest_b[kids].mean(0) if kids else bvh.rest_end(m[n]))
            if tip is None or end_b is None:
                continue
            ours_d = tip - sk.joints[n]
            theirs_d = basis @ (end_b - rest_b[m[n]])
        else:                                         # foot -> toe
            tip = sk.tips.get(f"toe_{s}")
            toe = _toe(bvh, s, m[n])
            if tip is None or toe is None:
                continue
            ours_d = tip - sk.joints[n]
            theirs_d = basis @ (toe - rest_b[m[n]])
        if np.linalg.norm(theirs_d) > 1e-9 and np.linalg.norm(ours_d) > 1e-9:
            align[n] = qarc(ours_d, theirs_d)

    # ---- world rotations of our joints (a joint the BVH lacks follows its parent)
    world: dict[str, np.ndarray] = {}
    for n in NAMES:
        if n in m:
            world[n] = qmul(conv_q(gb[:, m[n]]), align[n])
        else:
            world[n] = world[PARENT[n]] if PARENT[n] else np.tile([0.0, 0, 0, 1], (F, 1))
            notes.append(f"{n}: not in the BVH, follows its parent")
    hips_pos = pos_o[:, m["hips"]] * scale

    # ---- face +z on average over the range
    fw = qrot(world["hips"][f0:f1 + 1], np.array([0, 0, 1.0]))
    ang = math.atan2(float(fw[:, 0].mean()), float(fw[:, 2].mean()))
    yaw = qaxis(np.array([0, 1.0, 0]), -ang)
    for n in NAMES:
        world[n] = qmul(yaw, world[n])
    hips_pos = qrot(np.broadcast_to(yaw, (F, 4)), hips_pos)

    # ---- locals
    local = {}
    for n in NAMES:
        par = PARENT[n]
        local[n] = world[n] if par is None else qmul(qconj(world[par]), world[n])
        local[n] = _hemisphere(local[n])

    # ---- the cycle
    loops = False
    if loop is None or loop:
        cyc = find_cycle(local, hips_pos, bvh.fps, f0, f1, *(cycle or (0.45, 3.0)))
        if cyc is not None and (loop or cyc[2] < LOOP_OK):
            f0, f1, _ = cyc
            loops = True
        elif loop:
            notes.append("no clean cycle found: the whole range loops with a blended seam")
            loops = True
    elif end is None and (f1 - f0) / bvh.fps > max_shot:
        f0, f1 = action_window(local, bvh.fps, f0, f1, max_shot)
        notes.append(f"one-shot trimmed to {f0 / bvh.fps:.2f}-{f1 / bvh.fps:.2f} s (its main move)")
    sl = slice(f0, f1 + 1)
    local = {n: q[sl] for n, q in local.items()}
    hips_pos = hips_pos[sl]
    offset = hips_pos - hips_pos[0] * np.array([1, 0, 1])
    rest = sk.joints
    offset[:, 1] -= rest["hips"][1]                     # absolute BVH height -> offset, grounded below

    # ---- ground: the lowest foot over the clip reaches our rest foot height
    g, p = _our_fk(NAMES, PARENT, rest, local, offset)
    lows = np.minimum(p["foot_l"][:, 1], p["foot_r"][:, 1])
    ground_rest = min(rest["foot_l"][1], rest["foot_r"][1])
    offset[:, 1] += ground_rest - float(np.percentile(lows, 2))

    if plant_feet:
        local, offset, planted = _plant_feet(local, offset, rest, bvh.fps)
        if planted:
            notes.append(f"feet planted on {planted} contacts")

    # ---- travel
    dur = (len(offset) - 1) / bvh.fps
    travel = offset[-1] - offset[0]
    speed = float(np.hypot(travel[0], travel[2]) / max(dur, 1e-9))
    if in_place:
        u = np.linspace(0, 1, len(offset))[:, None]
        offset = offset - (offset[0] + u * (offset[-1] - offset[0])) * np.array([1, 0, 1])
    if loops:
        local, offset = _close_loop(local, offset)
    return Retargeted(local, offset, bvh.fps, speed, notes), loops


def action_window(local, fps, f0, f1, max_s=2.5, lead=0.2):
    """(first, last) frame of the strongest burst of motion in [f0, f1], at most max_s long:
    the smoothed joint angular speed peaks, the window grows to where it falls below a quarter
    of the peak (or the cap), with `lead` seconds of lead-in."""
    names = [n for n in local if not n.startswith(("hips",))]
    q = np.stack([local[n][f0:f1 + 1] for n in names], 1)            # (F, J, 4)
    dots = np.abs(np.sum(q[1:] * q[:-1], -1)).clip(0, 1)
    speed = np.r_[0.0, (2 * np.arccos(dots)).sum(1)] * fps          # rad / s, all joints
    k = max(1, int(fps * 0.25))
    e = np.convolve(speed, np.ones(k) / k, mode="same")
    n, cap = len(e), max(2, int(max_s * fps))
    pk = int(np.argmax(e))
    lo = hi = pk
    floor = e[pk] * 0.25
    while hi - lo < cap and (lo > 0 or hi < n - 1):
        left = e[lo - 1] if lo > 0 else -1
        right = e[hi + 1] if hi < n - 1 else -1
        if max(left, right) < floor:
            break
        if left >= right:
            lo -= 1
        else:
            hi += 1
    lo = max(0, lo - int(lead * fps))
    hi = min(n - 1, hi + int(lead * fps))
    if hi - lo > cap:
        c = (lo + hi) // 2
        lo, hi = max(0, c - cap // 2), min(n - 1, c + cap // 2)
    return f0 + lo, f0 + hi


LOOP_OK = 0.06          # mean pose mismatch (1 - |q.q'| + velocity) below which a clip loops by itself


def find_cycle(local, hips_pos, fps, f0, f1, min_s=0.45, max_s=3.0):
    """(first frame, last frame, mismatch) of the best repeating stretch in [f0, f1]."""
    names = list(local)
    step = max(1, int(round(fps / 30)))
    idx = np.arange(f0, f1 + 1, step)
    if len(idx) < 8:
        return None
    Q = np.stack([local[n][idx] for n in names if n != "hips"], 1)        # (T, J, 4)
    hy = hips_pos[idx, 1]
    span = max(float(np.ptp(hy)), 1e-6)
    T = len(idx)
    sub = fps / step                                                        # samples per second
    pmin, pmax = max(4, int(min_s * sub)), min(T - 2, int(max_s * sub))
    if pmax <= pmin:
        return None
    best = None
    vel = np.diff(Q, axis=0, prepend=Q[:1])
    # how far the pose wanders inside a window: a still stretch matches itself at any period,
    # so a window must move about as much as the clip's liveliest stretch of that length
    D = (1 - np.abs(np.einsum("ijk,ljk->ilj", Q, Q))).mean(-1)              # (T, T)
    C = np.cumsum(D, axis=1)
    # weight joints by how much they move in the clip (legs in a sneak, arms in a dance):
    # a mean over all joints lets the still ones hide a half-stride mismatch
    mean_q = Q.mean(0)
    mean_q /= np.linalg.norm(mean_q, axis=-1, keepdims=True) + 1e-12
    var = (1 - np.abs(np.sum(Q * mean_q, -1))).mean(0)                      # (J,)
    w = var / (var.sum() + 1e-12)
    for P in range(pmin, pmax + 1):
        a, b = Q[:T - P], Q[P:]
        d = 1 - np.abs(np.sum(a * b, -1))                                    # (S, J)
        dv = np.linalg.norm(vel[:T - P] - vel[P:], axis=-1) * 0.5
        dh = np.abs(hy[:T - P] - hy[P:]) / span * 0.02
        cost = (d * w).sum(1) + dv.mean(1) + dh
        r = np.arange(T - P)
        spread = (C[r, r + P] - np.where(r > 0, C[r, r - 1], 0)) / (P + 1)
        cost = np.where(spread >= 0.5 * spread.max(), cost, np.inf)
        s = int(np.argmin(cost))
        c = float(cost[s])
        if best is None or c < best[2] * 0.85 - 0.003:   # a longer period must be clearly better
            best = (s, P, c)
    s, P, c = best
    return int(idx[s]), int(idx[s + P]), c


def _close_loop(local, offset):
    """Spread the last-to-first mismatch over the clip so the last key equals the first."""
    F = len(offset)
    u = np.linspace(0, 1, F)
    out = {}
    for n, q in local.items():
        delta = qmul(qconj(q[-1]), q[0])                    # q[-1] * delta = q[0]
        ident = np.tile([0.0, 0, 0, 1], (F, 1))
        corr = qslerp(ident, np.broadcast_to(delta, (F, 4)), u)
        out[n] = _hemisphere(qmul(q, corr))
    off = offset + (offset[0] - offset[-1]) * u[:, None]
    return out, off


def _two_bone(h, k, a, t):
    """New knee and the two world corrections (thigh, shin) so the ankle a reaches t."""
    l1 = np.linalg.norm(k - h)
    l2 = np.linalg.norm(a - k)
    d = float(np.clip(np.linalg.norm(t - h), abs(l1 - l2) + 1e-6, (l1 + l2) * 0.9995))
    dirv = (t - h) / max(np.linalg.norm(t - h), 1e-12)
    bend = (k - h) - dirv * ((k - h) @ dirv)
    if np.linalg.norm(bend) < 1e-9:
        bend = np.cross(dirv, [1.0, 0, 0])
    bend /= np.linalg.norm(bend)
    cos_a = np.clip((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1)
    knee = h + l1 * (cos_a * dirv + math.sqrt(1 - cos_a * cos_a) * bend)
    r1 = qarc(k - h, knee - h)
    shin_now = qrot(r1, a - k)
    r2 = qarc(shin_now, (h + dirv * d) - knee)
    return r1, r2


def _plant_feet(local, offset, rest, fps, speed_tol=0.25, height_tol=0.04):
    """Pin each foot while it is down and slow (world space, root motion included). Where a
    straight leg can't reach its pinned spot the hips come down a little."""
    from .autorig import NAMES, PARENT
    H = max(float(rest["head"][1]), 1e-6)
    g, p = _our_fk(NAMES, PARENT, rest, local, offset)
    F = len(offset)
    targets = {}                       # side -> (F, 3) target, (F,) weight
    planted = 0
    for s in "lr":
        foot = p[f"foot_{s}"]
        vel = np.gradient(foot, axis=0) * fps / H
        low = foot[:, 1] < foot[:, 1].min() + height_tol * H
        contact = low & (np.linalg.norm(vel[:, [0, 2]], axis=1) < speed_tol)
        tgt, wt = foot.copy(), np.zeros(F)
        k = 0
        while k < F:
            if not contact[k]:
                k += 1
                continue
            e = k
            while e + 1 < F and contact[e + 1]:
                e += 1
            if e - k >= 2:
                spot = foot[k:e + 1].mean(0)
                spot[1] = float(np.median(foot[k:e + 1, 1]))
                n = e - k + 1
                up = np.arange(n) + 1 if k > 0 else np.full(n, 9)           # no ease at the clip's ends
                down = n - np.arange(n) if e < F - 1 else np.full(n, 9)
                wt[k:e + 1] = np.minimum(1.0, np.minimum(up, down) / 3.0)
                tgt[k:e + 1] = spot
                planted += 1
            k = e + 1
        targets[s] = (foot + wt[:, None] * (tgt - foot), wt)
    if not planted:
        return local, offset, 0
    # hips drop where a leg would have to stretch
    drop = np.zeros(F)
    for s in "lr":
        tgt, wt = targets[s]
        reach = 0.995 * (np.linalg.norm(rest[f"thigh_{s}"] - rest[f"shin_{s}"])
                         + np.linalg.norm(rest[f"shin_{s}"] - rest[f"foot_{s}"]))
        h = p[f"thigh_{s}"]
        horiz2 = np.sum((tgt - h)[:, [0, 2]] ** 2, 1)
        dy = h[:, 1] - tgt[:, 1]
        ok = np.sqrt(np.maximum(reach ** 2 - horiz2, 0))
        drop = np.maximum(drop, np.where(wt > 0, np.maximum(dy - ok, 0), 0))
    if drop.any():
        k = np.ones(5) / 5                                   # no sudden dips
        drop = np.maximum(drop, np.convolve(np.pad(drop, 2, mode="edge"), k, "valid"))
        offset = offset.copy()
        offset[:, 1] -= drop
        g, p = _our_fk(NAMES, PARENT, rest, local, offset)
    out = {n: q.copy() for n, q in local.items()}
    for s in "lr":
        tgt, wt = targets[s]
        for f in np.nonzero(wt > 0)[0]:
            r1, r2 = _two_bone(p[f"thigh_{s}"][f], p[f"shin_{s}"][f], p[f"foot_{s}"][f], tgt[f])
            gt = qmul(r1, g[f"thigh_{s}"][f])
            gs = qmul(r2, qmul(r1, g[f"shin_{s}"][f]))
            parent = g[PARENT[f"thigh_{s}"]][f]
            out[f"thigh_{s}"][f] = qmul(qconj(parent), gt)
            out[f"shin_{s}"][f] = qmul(qconj(gt), gs)
            out[f"foot_{s}"][f] = qmul(qconj(gs), g[f"foot_{s}"][f])    # the foot keeps its world turn
    return {n: _hemisphere(q) for n, q in out.items()}, offset, planted


def to_clip(motion: Retargeted, name: str, loop: bool, keys: int = 25):
    """(name, times, poses, loop) for autorig._clips_json: keys evenly over the motion."""
    F = len(motion.offset)
    dur = (F - 1) / motion.fps
    times = np.linspace(0, dur, keys)
    fpos = times * motion.fps
    i0 = np.clip(np.floor(fpos).astype(int), 0, F - 1)
    i1 = np.clip(i0 + 1, 0, F - 1)
    u = fpos - i0
    rots = {n: qslerp(q[i0], q[i1], u) for n, q in motion.local.items()}
    off = motion.offset[i0] + (motion.offset[i1] - motion.offset[i0]) * u[:, None]
    if loop:                                    # exactly the same first and last key
        for n in rots:
            rots[n][-1] = rots[n][0]
        off[-1] = off[0]
    poses = [({n: rots[n][k] for n in rots}, off[k]) for k in range(keys)]
    return name, times, poses, loop


def retarget_bvh(bvh_path, rigged_glb, name: str, out_glb=None, *, start: float | None = None,
                 end: float | None = None, in_place: bool = True, loop: bool | None = None,
                 plant_feet: bool = True, keys: int = 25, replace: bool = True,
                 cycle: tuple[float, float] | None = None) -> Path:
    """Append a BVH motion to an auto-rigged humanoid .glb as clip `name`.
    start/end: seconds into the BVH. loop: True = cut the best cycle and close it, False =
    one-shot, None = loop if a clean cycle is found. in_place: remove steady travel (the
    speed is kept in the clip's extras). The rest of the file stays byte for byte."""
    from .motions import append_clips, read_skeleton
    kind, sk, _ = read_skeleton(rigged_glb)
    if kind != "humanoid":
        raise ValueError("Mocap is retargeted onto the humanoid rig only (this is a four-legged rig)")
    bvh = parse_bvh(bvh_path)
    motion, loops = retarget(bvh, sk, start=start, end=end, in_place=in_place, loop=loop,
                             plant_feet=plant_feet, cycle=cycle)
    clip = to_clip(motion, name, loops, keys)
    extras = {"mocap": Path(str(bvh_path)).name if bvh.path else "bvh", "speed": round(motion.speed, 5)}
    return append_clips(rigged_glb, [clip], out_glb, replace=replace, extras={name: extras})


# --------------------------------------------------------------------------- the mocap library

INDEX = Path(__file__).with_name("mocap_library.json")


def mocap_root() -> Path:
    """Where downloaded BVH files live: $PIXEL_MOCAP or ~/Pixel Pipeline/Mocap."""
    env = os.environ.get("PIXEL_MOCAP")
    if env:
        return Path(env)
    home = os.environ.get("PIXEL_HOME")
    return (Path(home) if home else Path.home() / "Pixel Pipeline") / "Mocap"


def library_index() -> dict:
    """The curated index: {"clips": {friendly name: {file, start, end, loop, ...}}, "sources": ...}.
    A mocap_library.json in mocap_root() adds or overrides entries."""
    data = json.loads(INDEX.read_text(encoding="utf-8"))
    user = mocap_root() / "mocap_library.json"
    if user.exists():
        extra = json.loads(user.read_text(encoding="utf-8"))
        data.setdefault("clips", {}).update(extra.get("clips", {}))
    return data


def resolve(item) -> dict:
    """A BVH path or a friendly library name -> {path, name, start, end, loop}."""
    p = Path(str(item))
    if p.suffix.lower() == ".bvh" and p.exists():
        return {"path": p, "name": re.sub(r"[^0-9a-z]+", "_", p.stem.lower()).strip("_"),
                "start": None, "end": None, "loop": None, "cycle": None}
    index = library_index()
    clips = index["clips"]
    key = str(item).strip().lower()
    entry = clips.get(key) or clips.get(key.replace("_", " "))
    if entry is None:
        raise ValueError(f"No mocap clip '{item}' (library: {', '.join(sorted(clips))})")
    path = mocap_root() / entry["file"]
    if not path.exists():
        raise FileNotFoundError(f"{item}: {path} is not downloaded yet (see the download plan: "
                                f"{entry.get('url', 'mocap_library.json')})")
    start = entry.get("start")
    if start is None:              # e.g. the CMU conversions' added T-pose first frame
        start = index.get("sources", {}).get(entry.get("source"), {}).get("skip_first")
    return {"path": path, "name": entry.get("clip", re.sub(r"[^0-9a-z]+", "_", key).strip("_")),
            "start": start, "end": entry.get("end"), "loop": entry.get("loop"),
            "cycle": tuple(entry["cycle"]) if entry.get("cycle") else None}

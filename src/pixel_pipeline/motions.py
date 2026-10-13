"""More animations for the auto-rigged skeletons (autorig.py humanoid, quadruped.py animal).

Every clip is procedural: poses made from the skeleton found in the model, so they fit any
character the rig fits. Humanoid clips are key poses blended with smoothstep (like the
built-in attack / hurt / death); animal clips pose the body and put the paws with the same
2-bone IK as walk and sit, so planted feet stay planted.

    humanoid  cast (both arms gather, rise, push forward: a spell or a staff),
              shoot (bow: one arm points forward, the other draws to the chin and lets go),
              block (arms up in front, braced), thrust (spear lunge), jump, wave, kneel,
              victory
    animal    bark, howl, jump, eat

Weapon-aware attacks: a clip can be added under another name, "attack=shoot" (the clip is
called attack, it moves like shoot), so a tactics game that plays "attack" shows the right
move for the weapon. actions_for_role("archer") gives the default clip list with that swap.

Arms the rig couldn't separate from the body (robes, capes: the rig notes say "hanging at
the side") are bound to the robe around them, so they move at 40% there: cast and shoot
still read, the cloth doesn't tear.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# name -> (seconds, loops)
HUMAN = {"cast": (1.1, False), "shoot": (1.3, False), "block": (0.9, False),
         "thrust": (0.85, False), "jump": (1.0, False), "wave": (1.6, False),
         "kneel": (1.0, False), "victory": (1.4, False)}
QUAD = {"bark": (0.9, False), "howl": (2.0, False), "jump": (1.0, False), "eat": (1.8, False)}
WEAPON_ATTACK = {"sword": "attack", "spear": "thrust", "bow": "shoot", "staff": "cast"}
ROLE_WEAPON = {"knight": "sword", "swordsman": "sword", "paladin": "sword", "rogue": "sword",
               "spearman": "spear", "archer": "bow", "mage": "staff", "cleric": "staff",
               "physician": "staff"}
WEAK_ARM = 0.4            # how much an arm "not separate" from a robe moves


def parse_actions(actions, known) -> list[tuple[str, str]]:
    """['idle', 'attack=shoot'] -> [('idle', 'idle'), ('attack', 'shoot')], checked
    against the known motion names."""
    out = []
    for a in actions:
        name, _, motion = str(a).partition("=")
        name, motion = name.strip(), (motion or name).strip()
        if motion not in known:
            raise ValueError(f"Unknown animation '{motion}' (have: {', '.join(known)})")
        out.append((name, motion))
    return out


def attack_for(role: str | None = None, weapon: str | None = None) -> str:
    """The motion that is this role's (or weapon's) attack: attack (sword slash), thrust,
    shoot or cast."""
    weapon = weapon or ROLE_WEAPON.get((role or "").lower(), "sword")
    return WEAPON_ATTACK.get(weapon, "attack")


def actions_for_role(role: str | None = None, extra=()) -> tuple[str, ...]:
    """Default humanoid clips with the attack swapped for the role's weapon (archer:
    'attack=shoot'), plus any extra clip names."""
    from .autorig import ACTIONS
    motion = attack_for(role)
    base = tuple(a if a != "attack" or motion == "attack" else f"attack={motion}" for a in ACTIONS)
    return base + tuple(e for e in extra if e not in base)


# --------------------------------------------------------------------------- humanoid

def _arc(a, b) -> np.ndarray:
    """Shortest rotation (quaternion x y z w) turning direction a onto direction b."""
    a = np.asarray(a, np.float64) / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(b, np.float64) / max(np.linalg.norm(b), 1e-12)
    d = float(a @ b)
    if d < -0.9999:                           # opposite: any perpendicular axis
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 0, 1])
        axis /= np.linalg.norm(axis)
        return np.array([*axis, 0.0])
    q = np.array([*np.cross(a, b), 1.0 + d])
    return q / np.linalg.norm(q)


def _turn(q, v):
    u, w = q[:3], q[3]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def aim_arm(m, s: str, upper, fore) -> dict:
    """Arm pose by where the bones point, in the chest's frame: (forward, up, out), out =
    away from the body on that arm's side. Returns {upper_arm_s, forearm_s: quaternion}
    for the key poses (arms of a robe move WEAK_ARM of the way)."""
    from .autorig import _slerp
    j = m.sk.joints
    side = 1.0 if s == "l" else -1.0

    def vec(d):
        f, u, o = d
        return np.array([side * o, u, f], np.float64)
    rest_u = j[f"forearm_{s}"] - j[f"upper_arm_{s}"]
    rest_f = j[f"hand_{s}"] - j[f"forearm_{s}"]
    qu = _arc(rest_u, vec(upper))
    qf = _arc(rest_f, _turn(_conj(qu), vec(fore)))
    if m.weak.get(s):
        ru = m._q(m._rest(f"upper_arm_{s}"))
        rf = m._q(m._rest(f"forearm_{s}"))
        qu, qf = _slerp(ru, qu, WEAK_ARM), _slerp(rf, qf, WEAK_ARM)
    return {f"upper_arm_{s}": qu, f"forearm_{s}": qf}


def _both(m, upper, fore):
    return {**aim_arm(m, "l", upper, fore), **aim_arm(m, "r", upper, fore)}


DOWN = ((0.05, -1, 0.2), (0.15, -1, 0.1))      # hanging, elbow a little bent


def human_pose(m, action: str, p: float):
    """(rotations, hips offset) of a humanoid motion at progress p (0-1); m = autorig._Moves."""
    w, o = m.w, "l" if m.w == "r" else "r"
    sw = -1 if w == "r" else 1          # spine twist that brings the weapon shoulder back
    H = m.H
    A = lambda s, up, fo: aim_arm(m, s, up, fo)          # noqa: E731
    if action == "wave":
        return _wave(m, w, p)
    if action == "cast":
        keys = [(0.0, {}, 0),
                (0.3, {"spine": [("x", -6)], "chest": [("x", -4)], "head": [("x", -8)],
                       **_both(m, (0.55, -0.7, 0.35), (0.85, 0.25, -0.55))}, 0),
                (0.55, {"spine": [("x", -10)], "head": [("x", -14)],
                        **_both(m, (0.3, 1, 0.35), (0.25, 1, 0.1))}, 0),
                (0.72, {"spine": [("x", 12)], "chest": [("x", 6)], "head": [("x", 4)],
                        **_both(m, (1, 0.05, 0.12), (1, 0.12, -0.12)),
                        "thigh_" + o: [("x", -20)], "shin_" + o: [("x", 15)],
                        "thigh_" + w: [("x", 10)]}, -0.015),
                (1.0, {}, 0)]
    elif action == "shoot":
        bow = A(o, (0.87, 0.08, 0.5), (0.87, 0.1, 0.5))       # points at the target
        twist = {"spine": [("y", 30 * sw)], "head": [("y", -28 * sw)]}
        keys = [(0.0, {}, 0),
                (0.22, {"spine": [("y", 20 * sw)], "head": [("y", -18 * sw)], **bow,
                        **A(w, (0.9, 0.0, -0.1), (0.85, 0.1, -0.5))}, 0),     # hand to the string
                (0.55, {**twist, **bow, **A(w, (-0.3, 0.1, 0.95), (0.6, 0.35, -0.72))}, 0),
                (0.7, {**twist, **bow, **A(w, (-0.4, 0.1, 0.9), (0.55, 0.35, -0.75))}, 0),
                (0.8, {**twist, **bow, **A(w, (-0.5, 0.1, 0.85), (-0.35, 0.25, 0.9))}, 0),  # let go
                (1.0, {}, 0)]
    elif action == "block":
        def brace(push):
            return {"spine": [("x", 10 + 4 * push)], "head": [("x", -6)],
                    **_both(m, (0.7 + 0.2 * push, -0.25, 0.5), (0.3, 0.65, -0.75)),
                    "thigh_l": [("x", -18)], "thigh_r": [("x", -18)],
                    "shin_l": [("x", 30)], "shin_r": [("x", 30)]}
        keys = [(0.0, {}, 0), (0.25, brace(0), -0.02), (0.42, brace(1), -0.02),
                (0.75, brace(0), -0.02), (1.0, {}, 0)]
    elif action == "thrust":
        lunge = {"spine": [("y", -15 * sw), ("x", 14)],
                 **A(w, (1, 0.0, 0.1), (1, 0.0, 0.0)), **A(o, (0.8, -0.3, 0.2), (1, 0.1, -0.35)),
                 "thigh_" + o: [("x", -40)], "shin_" + o: [("x", 35)],
                 "thigh_" + w: [("x", 20)], "shin_" + w: [("x", 5)]}
        keys = [(0.0, {}, 0),
                (0.3, {"spine": [("y", 20 * sw), ("x", -4)],
                       **A(w, (-0.5, -0.75, 0.3), (0.85, -0.2, -0.3)),
                       **A(o, (0.5, -0.65, 0.3), (0.9, 0.1, -0.35)), "thigh_" + w: [("x", 10)]}, 0),
                (0.52, lunge, (0, -0.05, 0.06)), (0.7, lunge, (0, -0.05, 0.06)),
                (1.0, {}, 0)]
    elif action == "jump":
        keys = [(0.0, {}, 0),
                (0.22, {"spine": [("x", 14)], "head": [("x", -8)],
                        "thigh_l": [("x", -55)], "thigh_r": [("x", -55)],
                        "shin_l": [("x", 95)], "shin_r": [("x", 95)],
                        **_both(m, (-0.6, -0.8, 0.2), (-0.5, -0.85, 0.15))}, -0.15),
                (0.45, {"spine": [("x", -5)], "thigh_l": [("x", -30)], "thigh_r": [("x", -30)],
                        "shin_l": [("x", 50)], "shin_r": [("x", 50)],
                        **_both(m, (0.3, 1, 0.35), (0.2, 1, 0.25))}, 0.12),
                (0.68, {"spine": [("x", 15)], "thigh_l": [("x", -35)], "thigh_r": [("x", -35)],
                        "shin_l": [("x", 60)], "shin_r": [("x", 60)],
                        **_both(m, (0.3, -0.9, 0.4), (0.6, -0.6, 0.3))}, -0.07),
                (1.0, {}, 0)]
    elif action == "kneel":
        j = m.sk.joints
        drop = (j["shin_" + w][1] - 0.03 * H) / H       # the knee comes down to the ground
        down = {"thigh_" + w: [("x", -2)], "shin_" + w: [("x", 88)],
                "thigh_" + o: [("x", -85)], "shin_" + o: [("x", 85)],
                "spine": [("x", 12)], "head": [("x", 6)],
                **A(o, (0.5, -0.85, 0.15), (0.75, -0.55, 0.0)), **A(w, *DOWN)}
        keys = [(0.0, {}, 0), (0.6, down, -drop), (1.0, down, -drop)]
    elif action == "victory":
        def up(k):
            return {**_both(m, (0.1, 1, 0.45 - 0.2 * k), (0.05, 1, 0.3 - 0.2 * k)),
                    "spine": [("x", -8 - 4 * k)], "head": [("x", -12 - 3 * k)]}
        keys = [(0.0, {}, 0), (0.3, up(0), 0), (0.45, up(1), 0.03), (0.6, up(0), 0),
                (0.75, up(1), 0.03), (1.0, up(0), 0)]
    else:
        raise ValueError(f"Unknown animation {action}")
    return m.interp(keys, p)


def _ramp(p, a, b):
    from .autorig import _smooth
    return _smooth(min(1.0, max(0.0, (p - a) / max(b - a, 1e-9))))


def _wave(m, w, p):
    """Raise the hand, wave three times, put it down."""
    from .autorig import _slerp
    e = _ramp(p, 0.0, 0.18) * (1 - _ramp(p, 0.82, 1.0))
    u = (p - 0.18) / 0.64
    wag = 0.45 * math.sin(2 * math.pi * 3 * u) if 0 < u < 1 else 0.0
    arm = aim_arm(m, w, (0.15, 0.45, 0.88), (0.15, 1, 0.1 + wag))
    rest = {n: m._q(m._rest(n)) for n in arm}
    rot = {n: _slerp(rest[n], q, e) for n, q in arm.items()}
    side = 1 if w == "l" else -1
    rot.update({"head": [("z", -side * 5 * e)], "spine": [("z", -side * 2 * e)]})
    return rot, np.zeros(3)


# --------------------------------------------------------------------------- animal

def quad_pose(q, action: str, p: float):
    """(local joint rotations, root offset) of an animal motion; q = quadruped._Poser."""
    from .quadruped import _I, _q_of, _rx
    Lg = q.leg_len
    if action == "bark":
        def pulse(c):
            return math.exp(-((p - c) / 0.07) ** 2)
        b = pulse(0.28) + pulse(0.62)
        lean = _ramp(p, 0, 0.15) * (1 - _ramp(p, 0.8, 1.0))
        body = {"chest": _rx(-3 * lean - 4 * b), "neck": _rx(-10 * lean - 10 * b),
                "head": _rx(-4 * lean - 18 * b), "tail1": _q_of([("x", -10 * lean)]),
                "tail2": _rx(-5 * lean)}
        off = np.array([0, -0.015 * Lg * b, 0])
        return q.pose(body, off, q.planted()), off
    if action == "howl":
        u = _ramp(p, 0.0, 0.25) * (1 - _ramp(p, 0.8, 1.0))
        shake = 2 * math.sin(2 * math.pi * 9 * p) * u if 0.3 < p < 0.75 else 0.0
        body = {"chest": _rx(-6 * u), "neck": _rx(-50 * u), "head": _rx(-25 * u + shake),
                "tail1": _rx(-15 * u), "tail2": _rx(-8 * u)}
        off = np.array([0, -0.02 * Lg * u, 0])
        return q.pose(body, off, q.planted()), off
    if action == "eat":
        u = _ramp(p, 0.0, 0.3) * (1 - _ramp(p, 0.8, 1.0))
        nib = math.sin(2 * math.pi * 5 * p) if 0.3 < p < 0.8 else 0.0
        body = {"chest": _rx(10 * u), "neck": _rx(65 * u + 5 * nib * u), "head": _rx(30 * u),
                "tail1": _q_of([("y", 10 * math.sin(2 * math.pi * 2 * p) * u)])}
        off = np.array([0, -0.06 * Lg * u, 0])
        return q.pose(body, off, q.planted()), off
    if action == "jump":
        return _quad_jump(q, p, _I, _rx)
    raise ValueError(f"Unknown animation {action}")


def _lerp_keys(keys, p):
    """keys [(t, value)] -> smoothstep blend at p."""
    from .autorig import _smooth
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t0 <= p <= t1:
            u = _smooth((p - t0) / max(t1 - t0, 1e-9))
            return a + (b - a) * u
    return keys[-1][1]


def _quad_jump(q, p, _I, _rx):
    """Crouch, leap (front up, legs tucked), land front first, settle."""
    Lg = q.leg_len
    y = _lerp_keys([(0, 0.0), (0.22, -0.12), (0.45, 0.32), (0.62, 0.28), (0.8, -0.06), (1.0, 0.0)], p)
    pitch = _lerp_keys([(0, 0.0), (0.22, 4.0), (0.42, -14.0), (0.62, 8.0), (0.8, 2.0), (1.0, 0.0)], p)
    tuck = _lerp_keys([(0, 0.0), (0.3, 0.0), (0.45, 1.0), (0.62, 0.7), (0.74, 0.0), (1.0, 0.0)], p)
    off = np.array([0, y * Lg, 0])
    body = {"hips": _rx(pitch), "neck": _rx(-0.6 * pitch), "tail1": _rx(-10 * tuck),
            "tail2": _rx(-6 * tuck)}
    feet = {}
    for leg in ("front_l", "front_r", "back_l", "back_r"):
        rest = q.j[f"{leg}_paw"]
        ground = rest[1]
        air = max(ground, rest[1] + off[1])
        front = leg.startswith("front")
        lift = (0.3 if front else 0.12) * Lg * tuck
        dz = (-0.06 if front else -0.18) * Lg * tuck
        paw = (60 if front else 30) * tuck
        feet[leg] = (air + lift, rest[2] + dz, paw)
    return q.pose(body, off, feet), off


# --------------------------------------------------------------------------- adding clips to a file

def _world_rest(doc) -> dict[str, np.ndarray]:
    """Joint name -> rest position (the auto-rigs are world-aligned: translations only)."""
    nodes = doc["nodes"]
    out: dict[str, np.ndarray] = {}
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def pos(i):
        t = np.asarray(nodes[i].get("translation", [0, 0, 0]), np.float64)
        return t + (pos(parent[i]) if i in parent else 0)
    for i, n in enumerate(nodes):
        out[n.get("name", f"node{i}")] = pos(i)
    return out


def rig_kind(doc) -> str:
    names = {n.get("name") for n in doc.get("nodes", [])}
    if {"upper_arm_l", "thigh_r", "hips"} <= names:
        return "humanoid"
    if {"front_l_upper", "back_r_paw", "hips"} <= names:
        return "quadruped"
    raise ValueError("Not an auto-rigged character (no humanoid or four-legged skeleton): "
                     "add_animations works on files made by autorig / quadruped.")


def read_skeleton(rigged):
    """(kind, skeleton, weapon hand) of an auto-rigged .glb: from the skeleton saved in the
    file, or (older files) rebuilt from the joints and the mesh."""
    from .anim3d import Gltf, Rig
    g = Gltf(rigged)
    doc = g.doc
    kind = rig_kind(doc)
    saved = next((s.get("extras", {}).get("pixel_pipeline_rig") for s in doc.get("skins", [])
                  if isinstance(s.get("extras"), dict)), None)
    if saved:
        joints = {k: np.asarray(v, np.float64) for k, v in saved["joints"].items()}
        tips = {k: np.asarray(v, np.float64) for k, v in saved["tips"].items()}
        if kind == "humanoid":
            from .autorig import Skeleton
            return kind, Skeleton(joints, tips, saved["height"], saved.get("notes", [])), \
                saved.get("weapon", "r")
        from .quadruped import QuadSkeleton
        sk = QuadSkeleton(joints, tips, saved["height"], saved.get("notes", []),
                          saved.get("yaw", 0.0), saved.get("head", "+z"), saved.get("length", 1.0),
                          saved.get("belly", 0.5), saved.get("pair_z", 0.0),
                          saved.get("split_x", {}), saved.get("has_tail", False),
                          np.asarray(saved.get("body", []), np.float64).reshape(-1, 3))
        return kind, sk, "r"
    # an older file: joints from the nodes, the rest from the mesh
    rest = _world_rest(doc)
    rig = Rig(rigged)
    verts = np.concatenate(rig.vertices(rig.worlds()))
    H = float(verts[:, 1].max())
    if kind == "humanoid":
        from .autorig import NAMES, Skeleton
        j = {n: rest[n] for n in NAMES}
        notes = []
        for s, sign in (("l", 1), ("r", -1)):      # autorig's "hanging at the side" fallback
            d = j[f"hand_{s}"] - j[f"upper_arm_{s}"]
            if abs(d[0] - sign * 0.035 * H) < 2e-3 * H and abs(d[1] + 0.32 * H) < 2e-3 * H:
                notes.append(f"arm {s}: not separate, hanging at the side")
        tips = {"head_top": np.array([j["head"][0], H * 0.995, j["head"][2]])}
        for s in "lr":
            tips[f"finger_{s}"] = j[f"hand_{s}"] + 0.5 * (j[f"hand_{s}"] - j[f"forearm_{s}"])
            tips[f"toe_{s}"] = np.array([j[f"foot_{s}"][0], 0.01 * H, j[f"foot_{s}"][2] + 0.07 * H])
        return kind, Skeleton(j, tips, H, notes), "r"
    from .quadruped import LEGS, NAMES, QuadSkeleton
    j = {n: rest[n] for n in NAMES}
    belly = float(min(j[f"{leg}_upper"][1] for leg in LEGS)) - 0.04 * H
    tips = {f"{leg}_toe": np.array([j[f"{leg}_paw"][0], 0.01 * H, j[f"{leg}_paw"][2] + 0.04 * H])
            for leg in LEGS}
    body = verts[verts[:, 1] > 0.9 * belly]
    if len(body) > 3000:
        body = body[np.random.default_rng(0).choice(len(body), 3000, replace=False)]
    pair_z = float(np.mean([j[f"{leg}_upper"][2] for leg in LEGS]))
    split = {p: float(np.mean([j[f"{p}_{s}_upper"][0] for s in "lr"])) for p in ("front", "back")}
    sk = QuadSkeleton(j, tips, H, [], 0.0, "+z", float(np.ptp(verts[:, 2])), belly, pair_z, split,
                      True, body)
    return kind, sk, "r"


def add_animations(rigged_glb, names, out_glb=None, *, replace: bool = False,
                   weapon_hand: str | None = None) -> Path:
    """Append clips to an auto-rigged character file (humanoid or four-legged).

    names: motion names (motions.HUMAN / QUAD or the built-in ones), 'attack=shoot' to add
    a clip called attack that moves like shoot. The mesh, skin and every clip already in
    the file stay byte for byte as they were; names the file already has are skipped
    unless replace=True (then the old clip is dropped and the new one appended).
    out_glb: where to write (default: the same file)."""
    import json
    from .anim3d import Gltf, GlbWriter
    src = Path(rigged_glb)
    out = Path(out_glb) if out_glb else src
    kind, sk, weapon = read_skeleton(src)
    if weapon_hand:
        weapon = "r" if weapon_hand.startswith("r") else "l"
    g0 = Gltf(src)
    doc = json.loads(json.dumps(g0.doc))
    if len(doc.get("buffers", [])) != 1 or "uri" in doc["buffers"][0]:
        raise ValueError(f"{src.name}: expected one binary buffer (a .glb made by the auto-rig)")
    if kind == "humanoid":
        from .autorig import JOINTS, all_actions, human_clips
        known = all_actions()
    else:
        from .quadruped import JOINTS, all_actions, quad_clips
        known = all_actions()
    wanted = parse_actions(names, known)
    have = [a.get("name") for a in doc.get("animations", [])]
    todo = []
    for name, motion in wanted:
        if name in have and not replace:
            continue
        if name in have:
            doc["animations"] = [a for a in doc["animations"] if a.get("name") != name]
        todo.append(f"{name}={motion}")
    if todo:
        clips = human_clips(sk, todo, weapon) if kind == "humanoid" else quad_clips(sk, todo)
        g = GlbWriter()
        g.doc = doc
        g.bin = bytearray(g0.buffers[0][:doc["buffers"][0]["byteLength"]])
        index = {n.get("name"): i for i, n in enumerate(doc["nodes"])}
        joint_names = [n for n, _ in JOINTS]
        from .autorig import _clips_json
        doc.setdefault("animations", [])
        doc["animations"] += _clips_json(g, clips, joint_names, index, joint_names[0], sk.joints)
        tmp = out.with_name(out.name + ".tmp")
        g.save(tmp)
        tmp.replace(out)
    elif out != src:
        out.write_bytes(src.read_bytes())
    return out

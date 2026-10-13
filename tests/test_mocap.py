"""mocap.py: BVH parsing and retargeting onto the auto-rigged humanoid (synthetic BVH files)."""
import json
import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from test_autorig import humanoid  # noqa: E402

from pixel_pipeline.anim3d import Gltf, Rig  # noqa: E402
from pixel_pipeline.autorig import auto_rig  # noqa: E402
from pixel_pipeline.mocap import (BvhJoint, find_cycle, library_index, match_joints,  # noqa: E402
                                  parse_bvh, qaxis, qmul, qrot, resolve, retarget_bvh, write_bvh)

ROT = ["Zrotation", "Yrotation", "Xrotation"]


def _skeleton(prefix="", names=None):
    """A CMU-style skeleton in a T-pose, facing +x (left = -z), y up, ~cm units."""
    n = names or {}

    def nm(k):
        return prefix + n.get(k, k)
    spec = [("Hips", None, (0, 0, 0)), ("LHipJoint", "Hips", (0, 0, 0)), ("LeftUpLeg", "LHipJoint", (0, -6, -9)),
            ("LeftLeg", "LeftUpLeg", (0, -42, 0)), ("LeftFoot", "LeftLeg", (0, -40, 0)),
            ("LeftToeBase", "LeftFoot", (12, -6, 0)),
            ("RHipJoint", "Hips", (0, 0, 0)), ("RightUpLeg", "RHipJoint", (0, -6, 9)),
            ("RightLeg", "RightUpLeg", (0, -42, 0)), ("RightFoot", "RightLeg", (0, -40, 0)),
            ("RightToeBase", "RightFoot", (12, -6, 0)),
            ("LowerBack", "Hips", (0, 4, 0)), ("Spine", "LowerBack", (0, 10, 0)), ("Spine1", "Spine", (0, 10, 0)),
            ("Neck", "Spine1", (0, 14, 0)), ("Neck1", "Neck", (0, 4, 0)), ("Head", "Neck1", (0, 6, 0)),
            ("LeftShoulder", "Spine1", (0, 10, -4)), ("LeftArm", "LeftShoulder", (0, 0, -14)),
            ("LeftForeArm", "LeftArm", (0, 0, -28)), ("LeftHand", "LeftForeArm", (0, 0, -25)),
            ("RightShoulder", "Spine1", (0, 10, 4)), ("RightArm", "RightShoulder", (0, 0, 14)),
            ("RightForeArm", "RightArm", (0, 0, 28)), ("RightHand", "RightForeArm", (0, 0, 25))]
    joints, index = [], {}
    for name, parent, off in spec:
        p = index[parent] if parent else -1
        j = BvhJoint(nm(name), p, np.array(off, float),
                     (["Xposition", "Yposition", "Zposition"] if p < 0 else []) + ROT)
        index[name] = len(joints)
        if p >= 0:
            joints[p].children.append(len(joints))
        joints.append(j)
    ends = {"LeftToeBase": (4, 0, 0), "RightToeBase": (4, 0, 0), "Head": (0, 16, 0),
            "LeftHand": (0, 0, -10), "RightHand": (0, 0, 10)}
    for k, e in ends.items():
        joints[index[k]].end = np.array(e, float)
    return joints, index


A, DUTY, L = 20.0, 0.6, 82.0          # thigh swing, stance share, hip-to-ankle


def _thigh(p):
    """Thigh angle (degrees, + = forward) at cycle phase p: linear back in stance, eased forward."""
    p %= 1.0
    if p < DUTY:
        return A - 2 * A * p / DUTY
    s = (p - DUTY) / (1 - DUTY)
    return -A + 2 * A * s * s * (3 - 2 * s)


def cmu_walk(path, cycles=2.5, period=1.0, fps=60, prefix="", names=None, slip=1.0):
    """A walk along +x: planted stance feet (the thigh turns at constant speed while the hips move
    at the matching speed), knees bend in swing, arms down from the T-pose and swinging."""
    joints, ix = _skeleton(prefix, names)
    n = int(cycles * period * fps) + 1
    data = np.zeros((n, 3 + 3 * len(joints)))
    w = math.radians(2 * A / DUTY) / period            # stance angular speed (rad/s)
    v = L * w * 0.985 * slip                            # hips speed that keeps the stance foot still
    for f in range(n):
        t = f / fps
        p = t / period
        col = {}
        th = {"Left": _thigh(p), "Right": _thigh(p + 0.5)}
        stance = [s for s, ph in (("Left", p % 1), ("Right", (p + 0.5) % 1)) if ph < DUTY]
        height = min(L * math.cos(math.radians(th[s])) for s in stance) + 12
        col["root"] = (v * t, height, 0)
        for s in ("Left", "Right"):
            ph = (p + (0 if s == "Left" else 0.5)) % 1
            knee = -55 * math.sin(math.pi * (ph - DUTY) / (1 - DUTY)) ** 2 if ph >= DUTY else 0
            col[f"{s}UpLeg"] = (th[s], 0, 0)            # Zrotation = swing about the lateral axis
            col[f"{s}Leg"] = (knee, 0, 0)
            col[f"{s}Foot"] = (-th[s] - knee, 0, 0)      # sole stays flat-ish
            sign = 1 if s == "Left" else -1
            col[f"{s}Arm"] = (-15 * math.sin(2 * math.pi * p) * sign, 0, -75 * sign)   # down, then swing
            col[f"{s}ForeArm"] = (10, 0, 0)
        row = [*col["root"]]
        for j in joints:
            key = j.name[len(prefix):]
            for k, v_ in enumerate(col.get(key, (0, 0, 0))):
                row.append(v_)
        data[f] = row
    return write_bvh(path, joints, data, 1 / fps), v


@pytest.fixture(scope="module")
def rigged(tmp_path_factory):
    d = tmp_path_factory.mktemp("mocap")
    return auto_rig(humanoid(d / "h.glb"), d / "h.rigged.glb", actions=("idle",))


def test_parse_bvh_hierarchy_channels_and_frames(tmp_path):
    path, _ = cmu_walk(tmp_path / "w.bvh", cycles=1)
    b = parse_bvh(path)
    assert b.names[:3] == ["Hips", "LHipJoint", "LeftUpLeg"] and len(b.joints) == 25
    assert b.joints[0].channels[:3] == ["Xposition", "Yposition", "Zposition"]
    assert b.fps == pytest.approx(60, rel=1e-3) and len(b.frames) == 61
    assert np.allclose(b.joints[b.index("LeftToeBase")].end, [4, 0, 0])
    assert b.joints[b.index("Spine1")].children == [b.index("Neck"), b.index("LeftShoulder"),
                                                     b.index("RightShoulder")]
    m = match_joints(b)
    assert b.names[m["chest"]] == "Spine1" and b.names[m["spine"]] == "Spine"
    assert b.names[m["upper_arm_l"]] == "LeftArm" and b.names[m["foot_r"]] == "RightFoot"


def test_euler_order_follows_the_channels():
    """CHANNELS Zrotation Xrotation Yrotation = Rz * Rx * Ry (a vector turns by Ry first)."""
    text = """HIERARCHY
ROOT Bone
{
  OFFSET 0 0 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site
  {
    OFFSET 0 1 0
  }
}
MOTION
Frames: 1
Frame Time: 0.033333
90 90 0
"""
    b = parse_bvh(text)
    rot, _ = b.local()
    q = rot[0, 0]
    expect = qmul(qaxis(np.array([0, 0, 1.0]), math.pi / 2), qaxis(np.array([1.0, 0, 0]), math.pi / 2))
    assert np.allclose(np.abs(q @ expect), 1, atol=1e-9)
    # (0, 1, 0): Rx(90) -> (0, 0, 1), then Rz(90) leaves z alone
    assert np.allclose(qrot(q, np.array([0, 1.0, 0])), [0, 0, 1], atol=1e-9)


def _at(rig, clip, name, p):
    return rig.worlds(clip, clip.start + p * clip.duration)[rig.names.index(name)][:3, 3]


def test_walk_retargeted_onto_the_rig(rigged, tmp_path):
    bvh, v = cmu_walk(tmp_path / "walk.bvh")
    out = retarget_bvh(bvh, rigged.path, "walk", tmp_path / "w.glb")
    rig = Rig(out)
    clip = next(c for c in rig.clips if c.name == "walk")
    assert clip.loop and clip.duration == pytest.approx(1.0, abs=0.05)       # one cycle found
    j = rigged.skeleton.joints
    ps = np.linspace(0, 1, 41)
    fl = np.array([_at(rig, clip, "foot_l", p) for p in ps])
    fr = np.array([_at(rig, clip, "foot_r", p) for p in ps])
    hips = np.array([_at(rig, clip, "hips", p) for p in ps])
    # the actor walked along +x; here the character steps along +z (its front)
    dz_l, dz_r = fl[:, 2] - hips[:, 2], fr[:, 2] - hips[:, 2]
    assert np.ptp(dz_l) > 0.25 and np.ptp(dz_r) > 0.25
    assert np.corrcoef(dz_l, dz_r)[0, 1] < -0.6                              # opposite legs
    assert np.ptp(fl[:, 0]) < 0.06                                           # no sideways sweep
    # arms: from the T-pose rest, lowered to ~15 degrees -> hands hang below the A-pose rest
    hl = np.array([_at(rig, clip, "hand_l", p) for p in ps])
    assert hl[:, 1].mean() < j["hand_l"][1] - 0.1
    assert abs(hl[:, 0].mean() - j["upper_arm_l"][0]) < 0.25
    # loops close: last pose = first pose
    a, b = rig.worlds(clip, clip.start), rig.worlds(clip, clip.end)
    assert np.allclose(a[:, :3, 3], b[:, :3, 3], atol=1e-4)
    # in place, with the travel speed kept (leg-length scaled)
    assert np.ptp(hips[:, 2]) < 0.1 and np.ptp(hips[:, 0]) < 0.05
    extras = next(x for x in Gltf(out).doc["animations"] if x["name"] == "walk")["extras"]
    leg = np.mean([np.linalg.norm(j[f"thigh_{s}"] - j[f"shin_{s}"]) + np.linalg.norm(j[f"shin_{s}"] - j[f"foot_{s}"])
                   for s in "lr"])
    assert extras["speed"] == pytest.approx(v * leg / L, rel=0.15) and extras["mocap"] == "walk.bvh"
    # feet on the ground: the lowest foot is at the rest foot height
    low = np.minimum(fl[:, 1], fr[:, 1])
    assert abs(low.min() - j["foot_l"][1]) < 0.03
    # the old clips are untouched
    old = Gltf(rigged.path).doc["animations"]
    assert [a["name"] for a in Gltf(out).doc["animations"]] == [a["name"] for a in old] + ["walk"]


@pytest.mark.parametrize("plant", [True, False])
def test_feet_stay_planted_with_root_motion(rigged, tmp_path, plant):
    """The actor's feet slide (hips 30% slower than the stride): planting pins them."""
    bvh, _ = cmu_walk(tmp_path / "walk.bvh", slip=0.7)
    out = retarget_bvh(bvh, rigged.path, "walk", tmp_path / "w.glb", in_place=False, loop=False,
                       keys=121, plant_feet=plant)
    rig = Rig(out)
    clip = next(c for c in rig.clips if c.name == "walk")
    H = rigged.skeleton.height
    ps = np.linspace(0, 1, 121)
    t = ps * clip.duration
    worst = 0.0
    for side, shift in (("l", 0.0), ("r", 0.5)):
        foot = np.array([_at(rig, clip, f"foot_{side}", p) for p in ps])
        phase = (t + shift) % 1.0                         # the BVH's stance: phase < 0.6
        stance = (phase > 0.08) & (phase < 0.52)
        cycle = np.floor(t + shift)
        for c in np.unique(cycle[stance]):
            sel = stance & (cycle == c)
            if sel.sum() < 8:
                continue
            xz = foot[sel][:, [0, 2]]
            worst = max(worst, np.linalg.norm(xz - xz.mean(0), axis=1).max())
    assert (worst < 0.005 * H) if plant else (worst > 0.02 * H)
    hips = np.array([_at(rig, clip, "hips", p) for p in ps])
    assert hips[-1, 2] - hips[0, 2] > 1.0                   # it really travels forward (+z)


def test_mixamo_names_and_a_one_shot_range(rigged, tmp_path):
    names = {"LeftToeBase": "LeftToeBase", "Spine1": "Spine2"}
    bvh, _ = cmu_walk(tmp_path / "mx.bvh", prefix="mixamorig:", names=names)
    b = parse_bvh(bvh)
    m = match_joints(b)
    assert b.names[m["hips"]] == "mixamorig:Hips" and b.names[m["chest"]] == "mixamorig:Spine2"
    out = retarget_bvh(bvh, rigged.path, "step", tmp_path / "s.glb", start=0.2, end=0.9, loop=False)
    clip = next(c for c in Rig(out).clips if c.name == "step")
    assert not clip.loop and clip.duration == pytest.approx(0.7, abs=0.02)


def test_find_cycle_picks_the_period():
    fps = 30
    t = np.arange(0, 4, 1 / fps)
    q = np.stack([np.zeros_like(t), np.sin(math.pi * t / 1.3) * 0.3, np.zeros_like(t),
                  np.ones_like(t)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    local = {"hips": q, "spine": q, "thigh_l": q}
    s, e, cost = find_cycle(local, np.zeros((len(t), 3)), fps, 0, len(t) - 1)
    assert (e - s) / fps == pytest.approx(2.6, abs=0.1) and cost < 0.02


def test_library_names_resolve_to_downloaded_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXEL_MOCAP", str(tmp_path))
    idx = library_index()
    assert len(idx["clips"]) >= 30 and "walk" in idx["clips"] and idx["sources"]
    for name, e in idx["clips"].items():
        assert e["file"].lower().endswith(".bvh") and e["url"].startswith("http"), name
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        resolve("walk")
    with pytest.raises(ValueError, match="No mocap clip"):
        resolve("moonwalk on the ceiling")
    entry = idx["clips"]["walk"]
    f = tmp_path / entry["file"]
    f.parent.mkdir(parents=True, exist_ok=True)
    cmu_walk(f)
    r = resolve("walk")
    assert r["path"] == f and r["name"] == entry.get("clip", "walk")
    # a user index next to the files adds clips
    (tmp_path / "mocap_library.json").write_text(json.dumps(
        {"clips": {"my walk": {"file": entry["file"], "url": "https://example.org", "loop": True}}}))
    assert resolve("my walk")["loop"] is True


def test_library_add_mocap(tmp_path, monkeypatch):
    from pixel_pipeline import library
    from pixel_pipeline.assets import Style
    monkeypatch.setattr(library, "style", lambda: Style(directions=4, person_height=32))
    ch = library.Character("Walker", "person", "normal", "", 1.0, library.root() / "walker")
    ch.folder.mkdir(parents=True)
    auto_rig(humanoid(tmp_path / "h.glb"), ch.rigged, actions=("idle",))
    library.render_sheets(ch.rigged, ch.sheet, "person", "normal", frames=4)
    ch.save()
    bvh, _ = cmu_walk(tmp_path / "stroll.bvh")
    library.add_mocap(ch, [bvh])
    names = [a.name for a in library.load_animated(ch).actions]
    assert names == ["idle", "stroll"]

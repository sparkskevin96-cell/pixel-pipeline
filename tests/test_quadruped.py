"""Quadruped auto-rig on a box dog built here (body, 4 legs, head, tail)."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from pixel_pipeline import settings_for  # noqa: E402
from pixel_pipeline.anim3d import Gltf, animated_turnaround, load_rig  # noqa: E402
from pixel_pipeline.cli import main  # noqa: E402
from pixel_pipeline.quadruped import ACTIONS, LEGS, LOOPS, NAMES, auto_rig_quadruped  # noqa: E402

BODY, LEG, HEAD, TAIL = (150, 90, 40), (60, 40, 30), (220, 180, 120), (250, 250, 250)


def part(extents, centre, rgb):
    m = trimesh.creation.box(extents=extents).subdivide().subdivide()   # vertices along the legs
    m.apply_translation(centre)
    m.visual.vertex_colors = np.tile([*rgb, 255], (len(m.vertices), 1)).astype(np.uint8)
    return m


def box_dog(path, head=-1, tail=True):
    """1.0 long along x, head at `head` * x (like the Hunyuan dog: -x), legs 0.42 tall."""
    parts = [part([1.0, 0.35, 0.4], [0, 0.55, 0], BODY),
             part([0.3, 0.25, 0.25], [head * 0.6, 0.8, 0], HEAD)]
    parts += [part([0.1, 0.42, 0.1], [sx * 0.38, 0.21, sz * 0.13], LEG) for sx in (1, -1) for sz in (1, -1)]
    if tail:
        parts.append(part([0.3, 0.05, 0.05], [-head * 0.62, 0.62, 0], TAIL))
    trimesh.util.concatenate(parts).export(path)
    return path


@pytest.fixture(scope="module")
def rigged(tmp_path_factory):
    d = tmp_path_factory.mktemp("dog")
    return auto_rig_quadruped(box_dog(d / "dog.glb"), d / "dog.rigged.glb")


def test_skeleton_of_a_box_dog(rigged):
    sk = rigged.skeleton
    j = sk.joints
    assert rigged.notes == [] and sk.head == "-x" and sk.yaw == pytest.approx(90)
    assert sk.has_tail and set(NAMES) <= set(j)
    # turned: head +z, the dog's left +x. Legs at z = +-0.38, x = +-0.13
    for leg in LEGS:
        pair, side = leg.split("_")
        paw = j[f"{leg}_paw"]
        assert np.sign(paw[0]) == (1 if side == "l" else -1) and abs(abs(paw[0]) - 0.13) < 0.04
        assert np.sign(paw[2]) == (1 if pair == "front" else -1) and abs(abs(paw[2]) - 0.38) < 0.05
        assert paw[1] < 0.15 and j[f"{leg}_upper"][1] > 0.37 > j[f"{leg}_lower"][1]
    assert j["head"][2] > j["chest"][2] > j["spine"][2] > j["hips"][2] > j["tail1"][2] > j["tail2"][2]
    assert sk.tips["head_tip"][2] > 0.7 and j["head"][1] > 0.7
    assert rigged.preview.shape[2] == 4


def test_head_override_and_other_end(tmp_path):
    res = auto_rig_quadruped(box_dog(tmp_path / "d.glb", head=1, tail=False), tmp_path / "d.rigged.glb",
                             actions=["idle"])
    assert res.skeleton.head == "+x" and not res.skeleton.has_tail
    forced = auto_rig_quadruped(tmp_path / "d.glb", tmp_path / "f.rigged.glb", actions=["idle"], head="+x")
    assert forced.skeleton.yaw == pytest.approx(-90)
    assert np.allclose(forced.skeleton.joints["front_l_paw"], res.skeleton.joints["front_l_paw"], atol=0.02)


def test_a_diagonal_body_is_turned_straight(tmp_path):
    # prompt-to-model animals come from a three-quarter picture: the body lies diagonally
    m = trimesh.load(box_dog(tmp_path / "d.glb"), force="mesh")
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(40), [0, 1, 0]))
    m.export(tmp_path / "diag.glb")
    res = auto_rig_quadruped(tmp_path / "diag.glb", tmp_path / "diag.rigged.glb", actions=["idle"])
    j = res.skeleton.joints
    assert res.skeleton.head == "-x" and any("turned it straight" in n for n in res.notes)
    for leg in LEGS:                    # same layout as the straight dog: legs +-0.38 along z
        pair, side = leg.split("_")
        paw = j[f"{leg}_paw"]
        assert np.sign(paw[0]) == (1 if side == "l" else -1) and abs(abs(paw[0]) - 0.13) < 0.05
        assert np.sign(paw[2]) == (1 if pair == "front" else -1) and abs(abs(paw[2]) - 0.38) < 0.06
    assert j["head"][2] > j["chest"][2] > j["hips"][2]


def test_legs_bind_to_their_own_leg(rigged):
    rig = load_rig([rigged.path])
    prim = rig.primitives[0]
    v, joints, weights = prim.positions, prim.joints, prim.weights
    skin_joints = rig.skins[0][0]
    names = [rig.names[i] for i in skin_joints]
    below = v[:, 1] < 0.3                     # leg boxes, under the body (bottom at 0.375)
    assert below.sum() > 100
    for k in np.nonzero(below)[0]:
        x, z = v[k, 0], v[k, 2]
        leg = ("front" if z > 0 else "back") + "_" + ("l" if x > 0 else "r")
        own = sum(w for j, w in zip(joints[k], weights[k]) if names[j].startswith(leg))
        assert own > 0.9, (k, v[k], [(names[j], w) for j, w in zip(joints[k], weights[k])])


def test_animations_and_gait(rigged):
    rig = load_rig([rigged.path])
    clips = {c.name: c for c in rig.clips}
    assert [c.name for c in rig.clips] == list(ACTIONS)
    assert [clips[n].loop for n in ACTIONS] == [LOOPS[n] for n in ACTIONS]
    assert not clips["sit"].loop and clips["walk"].loop and not clips["death"].loop
    rest = np.concatenate(rig.vertices(rig.worlds()))
    paw = {leg: (rest[:, 1] < 0.08) & ((rest[:, 2] > 0) == leg.startswith("front"))
           & ((rest[:, 0] > 0) == leg.endswith("_l")) for leg in LEGS}

    def verts(name, t):
        return np.concatenate(rig.vertices(rig.worlds(clips[name], t)))
    walk = clips["walk"]
    zs = {leg: [] for leg in LEGS}
    ys = []
    for t in np.linspace(0, walk.end, 17)[:-1]:
        v = verts("walk", t)
        ys.append(v[:, 1].min())
        for leg in LEGS:
            zs[leg].append(v[paw[leg], 2].mean())
    zs = {leg: np.array(z) - np.mean(z) for leg, z in zs.items()}
    for leg in LEGS:
        assert np.ptp(zs[leg]) > 0.08, leg                          # every paw steps
    for a, b, quarter in (("front_l", "front_r", 8), ("back_l", "back_r", 8), ("back_l", "front_l", 4)):
        # 4-beat walk: right side half a cycle after the left, front a quarter after back
        assert np.abs(zs[b] - np.roll(zs[a], quarter)).max() < 0.02, (a, b)
        assert np.abs(zs[b] - zs[a]).max() > 0.08, (a, b)        # ... so not together
    assert max(abs(y - rest[:, 1].min()) for y in ys) < 0.03          # feet stay on the ground
    assert np.abs(verts("idle", 0.0) - verts("idle", clips["idle"].end)).max() < 1e-3
    sat = verts("sit", clips["sit"].end)
    rear = sat[:, 2] < -0.3
    assert sat[rear, 1].max() < rest[rest[:, 2] < -0.3, 1].max() - 0.15   # rump down
    front = (sat[:, 2] > 0.3) & (sat[:, 1] < 0.3)
    assert abs(sat[front, 1].min() - rest[:, 1].min()) < 0.03            # front paws planted
    down = verts("lie_down", clips["lie_down"].end)
    assert down[:, 1].max() < rest[:, 1].max() - 0.25
    dead = verts("death", clips["death"].end)
    assert dead[:, 1].min() > -0.03 and dead[:, 1].max() < 0.7          # on its side


def test_keeps_vertex_colours_and_renders(rigged):
    g = Gltf(rigged.path)
    assert "COLOR_0" in g.doc["meshes"][0]["primitives"][0]["attributes"]
    anim, _ = animated_turnaround([rigged.path], settings_for("animal", width=32, height=32), 4, 3,
                                  ["walk", "sit"], light=0, elevation=0)
    assert len(anim.directions) == 4 and anim.frames == 3 and [a.loop for a in anim.actions] == [True, False]
    side = anim.sprites[0][0][0].rgba                 # E: walking right, a side view
    cols = side[side[..., 3] > 0][:, :3].astype(int)
    assert (np.abs(cols - BODY).sum(1) < 60).mean() > 0.2
    assert (np.abs(cols - HEAD).sum(1) < 60).mean() > 0.03
    ys, xs = np.nonzero(side[..., 3])
    head_px = np.abs(side[..., :3].astype(int) - HEAD).sum(2) < 60
    assert np.nonzero(head_px)[1].mean() > xs.mean()   # head on the right in the E view
    frames = [anim.sprites[0][0][f].rgba for f in range(3)]
    assert any((frames[0] != f).any() for f in frames[1:])


def test_cli_rig_and_turnaround_animal(tmp_path, capsys):
    dog = box_dog(tmp_path / "dog.glb")
    assert main(["rig", str(dog), "--animal", "-o", str(tmp_path / "x.glb"), "--animations", "walk,sit",
                 "--preview", str(tmp_path / "x.png")]) == 0
    assert "skeleton + walk, sit" in capsys.readouterr().out and (tmp_path / "x.png").exists()
    out = tmp_path / "s" / "dog.png"
    assert main(["turnaround", str(dog), "--auto-rig", "-s", "animal", "--animations", "walk,happy",
                 "-o", str(out), "--directions", "4", "--frames", "2", "-W", "24", "-H", "24",
                 "--godot"]) == 0
    assert "2 animations x 4 directions x 2 frames" in capsys.readouterr().out
    assert (out.parent / "dog.rigged.glb").exists() and (out.parent / "dog.tres").exists()
    assert [c.name for c in load_rig([out.parent / "dog.rigged.glb"]).clips] == ["walk", "happy"]


def test_mcp_turnaround_auto_rig_animal(tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    dog = box_dog(tmp_path / "dog.glb")
    result = asyncio.run(server.call_tool("turnaround", {
        "source": [str(dog)], "out_png": str(tmp_path / "h.png"), "directions": 4, "auto_rig": True,
        "animal": True, "animations": ["sit"], "frames": 2, "width": 24, "height": 24}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    assert "sit (2 frames each)" in content[0].text and (tmp_path / "h.rigged.glb").exists()

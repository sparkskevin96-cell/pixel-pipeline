"""springs.py: capes, hair and tails swing and settle, baked into the clips."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from test_autorig import PANTS, SHIRT, SKIN, part  # noqa: E402
from test_quadruped import box_dog  # noqa: E402

from pixel_pipeline.anim3d import Gltf, Rig  # noqa: E402
from pixel_pipeline.autorig import auto_rig  # noqa: E402
from pixel_pipeline.quadruped import auto_rig_quadruped  # noqa: E402
from pixel_pipeline.springs import add_springs, has_springs, rebake  # noqa: E402

RED = (200, 30, 30)


def fine(m, size=0.03):
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, size)
    out = trimesh.Trimesh(v, f, process=False)
    out.visual.vertex_colors = np.tile(m.visual.vertex_colors[0], (len(v), 1))
    return out


def caped(path, cape=True, hair=False):
    """The box humanoid (1.8 tall, A-pose) with a thin cape hanging behind, shoulders to knees."""
    parts = [fine(part([0.42, 0.6, 0.26], [0, 1.15, 0], SHIRT), 0.06),
             fine(part([0.28, 0.3, 0.28], [0, 1.62, 0], SKIN), 0.06)]
    parts += [fine(part([0.15, 0.85, 0.16], [s * 0.11, 0.425, 0], PANTS), 0.06) for s in (1, -1)]
    for s in (1, -1):
        parts.append(fine(part([0.12, 0.62, 0.1], [s * 0.25, 1.13, 0], SKIN, s * 40, [s * 0.22, 1.42, 0]), 0.06))
    if cape:
        parts.append(fine(part([0.44, 0.95, 0.03], [0, 0.93, -0.3], RED)))
        parts.append(fine(part([0.44, 0.06, 0.2], [0, 1.42, -0.2], RED)))        # over the shoulders
    if hair:
        parts.append(fine(part([0.16, 0.4, 0.05], [0, 1.4, -0.2], (60, 40, 20))))
    trimesh.util.concatenate(parts).export(path)
    return path


@pytest.fixture(scope="module")
def knight(tmp_path_factory):
    d = tmp_path_factory.mktemp("cape")
    res = auto_rig(caped(d / "k.glb"), d / "k.rigged.glb", actions=("idle", "walk", "run", "death"))
    return res, add_springs(res.path, d / "k.springs.glb")


def _clip(rig, name):
    return next(c for c in rig.clips if c.name == name)


def _joint(rig, clip, name, t):
    return rig.worlds(clip, t)[rig.names.index(name)][:3, 3]


def _rest(rig, name):
    return rig.worlds()[rig.names.index(name)][:3, 3]


def test_cape_found_and_skinned(knight):
    res, out = knight
    doc = Gltf(out).doc
    names = [n["name"] for n in doc["nodes"]]
    chain = [n for n in names if n.startswith("spring_cape")]
    assert chain == ["spring_cape_0", "spring_cape_1", "spring_cape_2"]
    chest = next(n for n in doc["nodes"] if n["name"] == "chest")
    assert names.index("spring_cape_0") in chest["children"]
    rig = Rig(out)
    p = rig.primitives[0]
    joints = np.asarray(doc["skins"][0]["joints"])
    node_of = joints[p.joints]                                    # (n, 4) node indices
    spring = np.isin(node_of, [names.index(n) for n in chain])
    share = np.where(spring, p.weights, 0).sum(1)
    v = p.positions
    cape_low = (v[:, 2] < -0.2) & (v[:, 1] < 0.8)         # (the rig centres the model: cape at z -0.21)
    front = v[:, 2] > 0.1
    assert share[cape_low].mean() > 0.9 and share[front].max() < 0.05
    assert has_springs(out) and not has_springs(res.path)


def test_cape_trails_swings_and_loops(knight):
    res, out = knight
    rig = Rig(out)
    tip = "spring_cape_2"
    rest_rel = _rest(rig, tip) - _rest(rig, "hips")
    back = {}
    for name in ("walk", "run"):
        clip = _clip(rig, name)
        ts = np.linspace(clip.start, clip.end, 25)
        rel = np.array([_joint(rig, clip, tip, t) - _joint(rig, clip, "hips", t) for t in ts])
        back[name] = (rest_rel[2] - rel[:, 2]).mean()
        assert back[name] > 0.02, name                                   # flies out behind
        assert np.ptp(rel, 0).max() > 0.005, name                        # and moves
        a, b = rig.worlds(clip, clip.start), rig.worlds(clip, clip.end)
        assert np.allclose(a[:, :3, 3], b[:, :3, 3], atol=1e-4)          # still loops
    assert back["run"] > back["walk"]                                    # faster, further out


def test_cape_does_not_tear_or_pass_through(knight):
    res, out = knight
    rig, orig = Rig(out), Rig(res.path)
    rest = rig.vertices(rig.worlds())[0]
    p = rig.primitives[0]
    edges = np.concatenate([p.faces[:, [0, 1]], p.faces[:, [1, 2]], p.faces[:, [2, 0]]])
    L0 = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    cape = (rest[edges[:, 0], 2] < -0.17) & (L0 > 1e-6)
    for name in ("walk", "run", "death"):
        clip = _clip(rig, name)
        for t in np.linspace(clip.start, clip.end, 9):
            v = rig.vertices(rig.worlds(clip, t))[0]
            L = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
            o = orig.vertices(orig.worlds(_clip(orig, name), t))[0]       # the rig's own stretch
            Lo = np.linalg.norm(o[edges[:, 0]] - o[edges[:, 1]], axis=1)
            worse = (L / L0) / np.maximum(1.75, 1.2 * Lo / L0)      # never much worse than the rig
            # 1.1: since the spine stands straight over the knees (autorig._straighten_torso) the
            # cape's shoulder corner shares a little more with the swinging arm in a run
            assert worse[cape].max() < 1.1, (name, t)
            assert v[rest[:, 2] < -0.17, 1].min() > -0.02, (name, t)      # not through the floor
    clip = _clip(rig, "walk")                                             # stays behind the body
    for t in np.linspace(clip.start, clip.end, 9):
        w = rig.worlds(clip, t)
        chest = w[rig.names.index("chest")][:3, 3]
        for jn in ("spring_cape_1", "spring_cape_2"):
            assert w[rig.names.index(jn)][2, 3] < chest[2] - 0.05


def test_one_shot_settles_and_falls_with_the_body(knight):
    res, out = knight
    rig = Rig(out)
    clip = _clip(rig, "death")
    start = _joint(rig, clip, "spring_cape_2", clip.start)
    assert np.linalg.norm(start - _rest(rig, "spring_cape_2")) < 0.05    # starts settled
    assert _joint(rig, clip, "spring_cape_2", clip.end)[1] < 0.3          # lies down with the body


def test_rebake_and_new_clips(knight, tmp_path):
    res, out = knight
    from pixel_pipeline.motions import add_animations
    more = add_animations(out, ["jump"], tmp_path / "more.glb")
    rig = Rig(more)
    spring_nodes = {rig.names.index(n) for n in rig.names if n.startswith("spring_")}
    assert not {c.node for c in _clip(rig, "jump").channels} & spring_nodes   # no swing yet
    again = rebake(more, tmp_path / "again.glb")
    rig2 = Rig(again)
    assert {c.node for c in _clip(rig2, "jump").channels} >= spring_nodes
    assert [n for n in rig2.names if n.startswith("spring_")] == \
        [n for n in rig.names if n.startswith("spring_")]
    assert again.stat().st_size < more.stat().st_size * 1.2              # old data repacked


def test_no_cape_nothing_to_swing(tmp_path):
    res = auto_rig(caped(tmp_path / "p.glb", cape=False), tmp_path / "p.rigged.glb", actions=("idle",))
    with pytest.raises(ValueError, match="Nothing to swing"):
        add_springs(res.path, tmp_path / "x.glb")


def test_hair_chain(tmp_path):
    res = auto_rig(caped(tmp_path / "h.glb", cape=False, hair=True), tmp_path / "h.rigged.glb",
                   actions=("idle", "run"))
    out = add_springs(res.path, tmp_path / "hs.glb", parts=["hair"])
    doc = Gltf(out).doc
    names = [n["name"] for n in doc["nodes"]]
    assert "spring_hair_0" in names
    head = next(n for n in doc["nodes"] if n["name"] == "head")
    assert names.index("spring_hair_0") in head["children"]


def test_dog_tail(tmp_path):
    res = auto_rig_quadruped(box_dog(tmp_path / "dog.glb"), tmp_path / "dog.rigged.glb",
                             actions=("idle", "run", "happy"))
    out = add_springs(res.path, tmp_path / "dog.s.glb")
    rig = Rig(out)
    assert [n for n in rig.names if n.startswith("spring_")] == ["spring_tail_0", "spring_tail_1",
                                                                 "spring_tail_2"]
    run = _clip(rig, "run")
    tip = np.array([_joint(rig, run, "spring_tail_2", t) - _joint(rig, run, "hips", t)
                    for t in np.linspace(run.start, run.end, 13)])
    assert np.ptp(tip[:, 1]) > 0.005                                   # bounces with the gallop


def test_library_character_springs(tmp_path, monkeypatch):
    from pixel_pipeline import library
    from pixel_pipeline.assets import Style
    monkeypatch.setattr(library, "style", lambda: Style(directions=4, person_height=32))
    ch = library.Character("Caped", "person", "normal", "", 1.0, library.root() / "caped")
    ch.folder.mkdir(parents=True)
    auto_rig(caped(tmp_path / "c.glb"), ch.rigged, actions=("idle", "walk"))
    library.render_sheets(ch.rigged, ch.sheet, "person", "normal", frames=4)
    ch.save()
    library.add_character_springs(ch)
    assert has_springs(ch.rigged) and not (ch.folder / "rigged.new.glb").exists()
    assert [a.name for a in library.load_animated(ch).actions] == ["idle", "walk"]


def test_studio_has_the_swing_option():
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.model_studio import ModelStudio
    w = ModelStudio()
    assert w.swing.text() == "Capes/hair/tails swing" and not w.swing.isChecked()

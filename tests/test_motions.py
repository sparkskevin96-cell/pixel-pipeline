"""motions.py: more clips for the auto-rigs, added later to rigged files."""
import json

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from test_autorig import humanoid  # noqa: E402
from test_quadruped import box_dog  # noqa: E402

from pixel_pipeline import settings_for  # noqa: E402
from pixel_pipeline.anim3d import GlbWriter, Gltf, Rig, animated_turnaround  # noqa: E402
from pixel_pipeline.autorig import ACTIONS, _Moves, auto_rig  # noqa: E402
from pixel_pipeline.motions import (HUMAN, QUAD, actions_for_role, add_animations,  # noqa: E402
                                    attack_for, read_skeleton)
from pixel_pipeline.quadruped import auto_rig_quadruped  # noqa: E402


def clip_data(path):
    g = Gltf(path)
    return {a["name"]: [(g.accessor(s["input"]).tobytes(), g.accessor(s["output"]).tobytes())
                        for s in a["samplers"]] for a in g.doc["animations"]}


def joint_at(rig, clip, name, p):
    w = rig.worlds(clip, clip.start + p * clip.duration)
    return w[rig.names.index(name)][:3, 3]


@pytest.fixture(scope="module")
def rigged(tmp_path_factory):
    d = tmp_path_factory.mktemp("hum")
    return auto_rig(humanoid(d / "h.glb"), d / "h.rigged.glb", actions=("idle", "walk", "attack"))


def test_add_animations_keeps_old_clips_byte_for_byte(rigged, tmp_path):
    before = clip_data(rigged.path)
    out = add_animations(rigged.path, list(HUMAN), tmp_path / "more.glb")
    after = clip_data(out)
    assert list(after) == ["idle", "walk", "attack", *HUMAN]
    for name in before:
        assert after[name] == before[name]
    g0, g1 = Gltf(rigged.path), Gltf(out)
    assert g0.doc["meshes"] == g1.doc["meshes"] and g0.doc["skins"] == g1.doc["skins"]
    rig = Rig(out)
    loops = {c.name: c.loop for c in rig.clips}
    assert not any(loops[n] for n in HUMAN) and loops["walk"]
    # an existing name is skipped... unless replaced
    again = add_animations(out, ["cast"], tmp_path / "again.glb")
    assert list(clip_data(again)) == list(after)
    swapped = add_animations(out, ["attack=shoot"], tmp_path / "swap.glb", replace=True)
    data = clip_data(swapped)
    assert sorted(data) == sorted(after) and data["attack"] == after["shoot"]
    with pytest.raises(ValueError, match="Unknown animation 'moonwalk'"):
        add_animations(out, ["moonwalk"], tmp_path / "x.glb")


def test_motions_move_the_right_bones(rigged, tmp_path):
    out = add_animations(rigged.path, ["cast", "shoot", "kneel", "jump", "thrust"], tmp_path / "m.glb")
    rig = Rig(out)
    clips = {c.name: c for c in rig.clips}
    head = joint_at(rig, clips["cast"], "head", 0)[1]
    for s in "lr":                              # both hands over the head mid-cast
        assert joint_at(rig, clips["cast"], f"hand_{s}", 0.55)[1] > head
    # shoot: the bow hand (left, the weapon hand is right) points forward, the draw hand back
    bow = joint_at(rig, clips["shoot"], "hand_l", 0.6)
    draw = joint_at(rig, clips["shoot"], "hand_r", 0.6)
    assert bow[2] > 0.3 and draw[2] < bow[2] - 0.3
    hips0 = joint_at(rig, clips["kneel"], "hips", 0)[1]
    assert joint_at(rig, clips["kneel"], "hips", 1)[1] < hips0 - 0.3
    assert joint_at(rig, clips["kneel"], "shin_r", 1)[1] < 0.12        # the knee is down
    assert joint_at(rig, clips["jump"], "hips", 0.45)[1] > hips0 + 0.1
    assert joint_at(rig, clips["thrust"], "hand_r", 0.6)[2] > 0.3       # spear forward


def test_arms_hanging_at_the_side_move_gently(rigged):
    sk = rigged.skeleton
    weak = type(sk)(sk.joints, sk.tips, sk.height, ["arm l: not separate, hanging at the side"])
    full, soft = _Moves(sk, "r"), _Moves(weak, "r")
    assert soft.weak == {"l": True, "r": False}

    def angle(m, joint):
        q = m.quats("cast", 0.55)[0][joint]
        return 2 * np.degrees(np.arccos(min(1.0, abs(q[3]))))
    assert 0 < angle(soft, "upper_arm_l") < 0.6 * angle(full, "upper_arm_l")
    assert angle(soft, "upper_arm_r") == pytest.approx(angle(full, "upper_arm_r"))


def test_older_files_without_a_saved_skeleton(rigged, tmp_path):
    g = Gltf(rigged.path)
    doc = json.loads(json.dumps(g.doc))
    doc["skins"][0].pop("extras")
    w = GlbWriter()
    w.doc, w.bin = doc, bytearray(g.buffers[0][:doc["buffers"][0]["byteLength"]])
    old = w.save(tmp_path / "old.glb")
    kind, sk, _ = read_skeleton(old)
    assert kind == "humanoid" and sk.notes == []
    for n, p in rigged.skeleton.joints.items():
        assert np.allclose(sk.joints[n], p, atol=1e-5)
    assert abs(sk.height - rigged.skeleton.height) < 0.03
    out = add_animations(old, ["wave"], tmp_path / "old2.glb")
    assert "wave" in clip_data(out)


def test_weapon_attacks_by_role(tmp_path):
    assert attack_for("archer") == "shoot" and attack_for("spearman") == "thrust"
    assert attack_for("mage") == "cast" and attack_for("knight") == "attack"
    acts = actions_for_role("archer", extra=("block",))
    assert "attack=shoot" in acts and acts[-1] == "block" and len(acts) == len(ACTIONS) + 1
    res = auto_rig(humanoid(tmp_path / "a.glb"), tmp_path / "a.rigged.glb", actions=acts)
    assert res.clips == ["idle", "walk", "run", "attack", "hurt", "death", "block"]


def test_animal_motions(tmp_path):
    res = auto_rig_quadruped(box_dog(tmp_path / "dog.glb"), tmp_path / "dog.rigged.glb",
                             actions=("idle", "walk"))
    out = add_animations(res.path, list(QUAD))
    rig = Rig(out)
    clips = {c.name: c for c in rig.clips}
    assert set(QUAD) <= set(clips)
    head0 = joint_at(rig, clips["eat"], "head", 0)[1]
    assert joint_at(rig, clips["eat"], "head", 0.5)[1] < head0 - 0.1       # head down to eat
    assert joint_at(rig, clips["howl"], "head", 0.5)[1] > head0 + 0.01     # nose up
    for leg in ("front_l", "back_r"):                                       # paws stay planted
        assert abs(joint_at(rig, clips["eat"], f"{leg}_paw", 0.5)[1]
                   - joint_at(rig, clips["eat"], f"{leg}_paw", 0)[1]) < 0.02
    hips0 = joint_at(rig, clips["jump"], "hips", 0)[1]
    assert joint_at(rig, clips["jump"], "hips", 0.5)[1] > hips0 + 0.1
    assert joint_at(rig, clips["jump"], "front_l_paw", 0.5)[1] > 0.15      # off the ground


def test_new_clips_keep_the_old_frames(rigged, tmp_path):
    """frame_with=<the old clips>: adding clips leaves the old sprites exactly as they were."""
    out = add_animations(rigged.path, ["victory"], tmp_path / "v.glb")
    s = settings_for("person", width=24, height=32, colors=10)
    kw = dict(points=40_000, size=128)
    old, _ = animated_turnaround([rigged.path], s, 4, 2, ["idle", "walk"], **kw)
    new, _ = animated_turnaround([out], s, 4, 2, ["idle", "walk", "victory"],
                                 frame_with=["idle", "walk"], **kw)
    for a in range(2):
        for d in range(4):
            for f in range(2):
                assert np.array_equal(old.sprites[a][d][f].rgba, new.sprites[a][d][f].rgba)
    assert len(new.sprites) == 3


def test_library_character_gets_more_animations_later(tmp_path, monkeypatch):
    from pixel_pipeline import library
    from pixel_pipeline.assets import Style
    small = Style(directions=4, person_height=32)
    monkeypatch.setattr(library, "style", lambda: small)
    ch = library.Character("Hero", "person", "normal", "archer", 1.0,
                           library.root() / "hero")
    ch.folder.mkdir(parents=True)
    auto_rig(humanoid(tmp_path / "h.glb"), ch.rigged, actions=("idle", "walk"))
    library.render_sheets(ch.rigged, ch.sheet, "person", "normal", frames=8)
    ch.save()
    before_rig = clip_data(ch.rigged)
    before = library.load_animated(ch)
    seen = []
    library.add_character_animations(ch, ["shoot", "wave"], progress=seen.append)
    after = library.load_animated(ch)
    assert [a.name for a in after.actions] == ["idle", "walk", "shoot", "wave"]
    assert all(clip_data(ch.rigged)[n] == before_rig[n] for n in before_rig)
    for a in range(2):                                    # old clips' frames unchanged
        for d in range(4):
            for f in range(8):
                assert np.array_equal(before.sprites[a][d][f].rgba, after.sprites[a][d][f].rgba)
    assert ch.thumb.exists() and (ch.folder / "sheet_n.png").exists() and seen
    assert not (ch.folder / "rigged.new.glb").exists()

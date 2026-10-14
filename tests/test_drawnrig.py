"""Drawn characters bent by the skeleton (drawnrig.py): the warp maths on the hand-built robot of
test_anim3d (no GPU), and the whole flow against the fake ComfyUI."""
import json

import numpy as np
import pytest

from pixel_pipeline import drawnrig as dr
from pixel_pipeline import settings_for
from pixel_pipeline.anim3d import load_rig, write_animated
from pixel_pipeline.assets import Style
from test_anim3d import write_robot
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

W, H = 64, 48
PAL = np.array([[200, 30, 30], [30, 200, 30], [30, 30, 200], [230, 230, 60]])


@pytest.fixture
def robot(tmp_path):
    rig = load_rig([write_robot(tmp_path / "robot.glb")])
    body = dr.Body(rig, 60_000)
    rest = dr.rest_pose(body)
    poses = dr.clip_poses(body, rig.clips, 5)
    cams = dr.framing(body, [p for row in poses for p in row] + [rest], ["E", "S", "W", "N"], 30.0,
                      W, H)
    return rig, body, rest, poses, cams


def checker_key(view):
    ys, xs = np.mgrid[0:H, 0:W]
    return np.where(view.mask, (xs // 3 + ys // 3) % 3, -1).astype(np.int32)


def test_rest_pose_reproduces_the_key_drawing_exactly(robot):
    rig, body, rest, poses, cams = robot
    for cam in cams:
        view = dr.rest_view(body, cam, rest)
        key = checker_key(view)
        drawn = dr.glue(body, view, cam, rest, key, PAL)
        dr.add_hidden(drawn, body, view, rest, np.full(len(body.bind), 3, np.int32))
        assert len(drawn.hidden)                       # the box's far side is behind it
        out, normals = dr.bend(drawn, body, rest)
        assert np.array_equal(out, key)                # every pixel where it was drawn
        assert normals.shape == (H, W, 3)
        assert np.allclose(np.linalg.norm(normals[key >= 0], axis=-1), 1, atol=1e-6)


def test_pixels_the_drawing_added_outside_the_model_stay_put(robot):
    rig, body, rest, poses, cams = robot
    cam = cams[1]
    view = dr.rest_view(body, cam, rest)
    key = checker_key(view)
    ys, xs = np.nonzero(view.mask)
    key[ys.min() - 2, xs.min():xs.min() + 4] = 3         # a plume drawn above the model
    drawn = dr.glue(body, view, cam, rest, key, PAL)
    out, _ = dr.bend(drawn, body, rest, fill=False)
    assert np.array_equal(out, key)


def test_raising_the_arm_moves_only_the_arm_pixels(robot):
    rig, body, rest, poses, cams = robot
    cam = cams[1]                                      # S: the arm sticks out to the right
    view = dr.rest_view(body, cam, rest)
    arm = view.weights[..., 1] > 0.5                   # skin joint 1 = 'arm'
    key = np.where(view.mask, np.where(arm, 2, 1), -1).astype(np.int32)
    drawn = dr.glue(body, view, cam, rest, key, PAL)
    wave = poses[0][2]                                 # 'wave' at its top: arm up 90 degrees
    out, _ = dr.bend(drawn, body, wave, fill=False)
    edge, _, _ = cam.project(np.array([0.15, 0.0, 0.0]) - wave.shift)  # left of the raised arm
    body_px = (key == 1) & (np.arange(W)[None, :] < edge)
    assert (out[body_px] == 1).mean() > 0.98           # the body stays as drawn
    ys, xs = np.nonzero(key == 2)
    up_ys, up_xs = np.nonzero(out == 2)
    assert len(up_ys) > 0.6 * len(ys)
    assert up_ys.mean() < ys.mean() - 3                # the arm's pixels went up
    assert np.ptp(up_ys) > np.ptp(ys) + 4              # and it now stands upright
    assert np.ptp(up_xs) < np.ptp(xs)
    assert set(np.unique(out)) <= {-1, 1, 2}           # no new colours


def test_drop_fragments():
    idx = np.full((10, 10), -1, np.int32)
    idx[2:8, 2:8] = 1
    idx[4, 4] = -1                                     # a pinhole
    idx[0, 9] = 2                                      # a torn-off pixel
    out = dr.drop_fragments(idx)
    assert out[0, 9] == -1 and out[4, 4] == 1 and (out[2:8, 2:8] == 1).all()


def test_drawn_animation_with_the_fake_comfyui(comfy, tmp_path):  # noqa: F811
    fake, client = comfy
    rigged = write_robot(tmp_path / "robot.glb")
    s = settings_for("person", width=W, height=H, colors=12)
    s.outline_style = "selout"
    keys = tmp_path / "keys"
    anim, info = dr.drawn_animation(rigged, "a robot with a shield", s, directions=4, frames=3,
                                    animations=["wave", "fall"], seed=7, style=Style(),
                                    client=client, folder=keys, points=40_000, retries=0)
    assert len(fake.prompts) == 4 * len(dr.CANDIDATES)  # candidates per direction
    ks = [n for n in fake.prompts[0].values() if n["class_type"] == "KSampler"][0]["inputs"]
    assert ks["seed"] == 7 and 0 < ks["denoise"] < 1   # img2img over the model's render
    first = fake.prompts[0]                            # S, the reference: plain img2img
    assert not any(n["class_type"] == "LoadImageMask" for n in first.values())
    for wf in fake.prompts[len(dr.CANDIDATES):]:       # the rest: inpaint beside the reference
        assert any(n["class_type"] == "LoadImageMask" for n in wf.values())
        img = next(n for n in wf.values() if n["class_type"] == "LoadImage")["inputs"]["image"]
        w, h = fake.uploads[img].size
        assert w in (2 * h, 3 * h)                     # [reference(s) | this view]
    assert set(info["keys_report"]) == {"E", "S", "W", "N"}
    assert info["keys_report"]["N"]["refs"] == ["S", "E"]   # back view: front + side
    texts = [n["inputs"]["text"] for wf in fake.prompts for n in wf.values()
             if n["class_type"] == "CLIPTextEncode" and n.get("_meta", {}).get("title") == "Prompt"]
    assert any("facing right" in t for t in texts) and any("back view" in t for t in texts)
    assert any("turnaround" in t for t in texts)
    assert not any("shield" in t for t in texts)       # words never add held things
    assert anim.directions == ["E", "S", "W", "N"] and anim.frames == 3
    assert [a.name for a in anim.actions] == ["wave", "fall"]
    pal = anim.sprites[0][0][0].palette
    for sp in anim.flat():
        assert sp.rgba.shape == (H, W, 4) and sp.normal is not None
        assert sp.palette is pal                       # one palette, no new colours
    assert (keys / "drawn_S_o60.png").exists() and (keys / "key_S.png").exists()
    assert set(info["ious"]) == {"E", "S", "W", "N"}
    written = write_animated(anim, tmp_path / "sheet.png", aseprite=True, godot=True, normals=True)
    meta = json.loads((tmp_path / "sheet.json").read_text())
    assert {t["name"] for t in meta["meta"]["frameTags"]} >= {"wave_S", "fall_N"}
    assert any(p.name == "sheet_n.png" for p in written)
    (keys / "keys.json").write_text(json.dumps({"camera": info["camera"]}), encoding="utf-8")
    again, _ = dr.drawn_animation(rigged, "a robot", s, directions=4, frames=3,
                                  animations=["wave", "fall"], seed=7, style=Style(),
                                  client=client, folder=keys, points=40_000, retries=0)
    assert len(fake.prompts) == 4 * len(dr.CANDIDATES)  # drawings reused: no GPU
    assert np.array_equal(again.sprites[0][1][0].rgba, anim.sprites[0][1][0].rgba)


def test_library_render_sheets_drawn(comfy, tmp_path, monkeypatch):  # noqa: F811
    from pixel_pipeline import library
    fake, client = comfy
    from pixel_pipeline import generate
    monkeypatch.setattr(generate, "ComfyClient", lambda *a, **k: client)
    monkeypatch.setattr(dr, "_gpu_temperature", lambda: None)
    fake.history_delay = 0
    monkeypatch.setattr(library, "style", lambda: Style(directions=4))
    rigged = write_robot(tmp_path / "robot.glb")
    out = tmp_path / "ch" / "sheet.png"
    out.parent.mkdir()
    written = library.render_sheets(rigged, out, "person", "normal", frames=2, drawn=True,
                                    prompt="a robot")
    drew = len(fake.prompts)
    assert drew >= 4                                   # one per direction (+ redraws)
    names = {p.name for p in written}
    assert {"sheet.png", "sheet.json", "sheet_n.png", "sheet.aseprite", "sheet.tres"} <= names
    cam = json.loads((out.parent / "keys" / "keys.json").read_text())["camera"]
    assert (cam["width"], cam["height"], cam["directions"]) == (96, 64, 4)
    ch = library.Character("Robot", folder=out.parent)
    assert library.is_drawn(ch)
    library.render_sheets(rigged, out, "person", "normal", frames=2, drawn=True, prompt="a robot")
    assert len(fake.prompts) == drew                   # kept drawings, same scale


def test_model_words_drop_held_things():
    assert dr.model_words("an armored knight in steel plate armour with a red cape, sword and "
                          "shield") == "an armored knight in steel plate armour with a red cape"
    assert dr.model_words("a wizard holding a glowing staff, blue robes") == "a wizard, blue robes"
    assert dr.model_words("a swordsman in blue") == "a swordsman in blue"


def test_consistency_scores():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, :4] = (200, 30, 30)
    rgb[:, 4:] = (30, 30, 200)
    m = np.ones((8, 8), bool)
    assert dr.colour_distance(rgb, m, rgb, m) == 0
    other = np.full_like(rgb, (30, 200, 30))
    assert dr.colour_distance(rgb, m, other, m) == pytest.approx(1.0)
    flat = np.full_like(rgb, 128)
    assert dr.sharpness(rgb, m) > dr.sharpness(flat, m) == 0


def test_lying_down_uses_the_model_frame(robot):
    rig, body, rest, poses, cams = robot
    torso = rig.names.index("root")
    fall = [c.name for c in rig.clips].index("slide")
    assert dr.turned_away(rest, rest, cams[1], torso) == pytest.approx(0, abs=1e-6)
    assert dr.turned_away(rest, poses[fall][2], cams[1], torso) < 1   # sliding stays upright
    idx, normals = dr.model_frame(body, rest, cams[1], np.full(len(body.bind), 2, np.int32))
    view = dr.rest_view(body, cams[1], rest)
    assert dr.iou(idx >= 0, view.mask) > 0.9 and set(np.unique(idx)) <= {-1, 2}


def test_majority_calms_speckle():
    idx = np.full((6, 6), 1, np.int32)
    idx[2, 3] = 2                                      # one stray colour
    idx[0, 0] = -1
    out = dr.majority(idx)
    assert out[2, 3] == 1 and out[0, 0] == -1 and (out[idx >= 0] == 1).all()

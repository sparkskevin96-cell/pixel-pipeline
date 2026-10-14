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
    per_dir = 3 * len(dr.CANDIDATES) + len(dr.BACK_CANDIDATES)   # N: one more try
    assert len(fake.prompts) == per_dir                # candidates per direction
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
    negs = [n["inputs"]["text"] for wf in fake.prompts for n in wf.values()
            if n["class_type"] == "CLIPTextEncode" and "Negative" in n.get("_meta", {}).get("title", "")]
    assert negs and all("shield" in t for t in negs)   # the model has none: kept out
    assert info["absent"] == list(dr.SHIELD_WORDS)
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
    assert len(fake.prompts) == per_dir                # drawings reused: no GPU
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


# --------------------------------------------------------------------------- v3: any subject

def _view(mask, colour, bones, nb=3):
    """A hand-made RestView (no rig): weights one-hot by bone, flat depth."""
    H, W = mask.shape
    w = np.zeros((H, W, nb))
    w[np.arange(H)[:, None], np.arange(W)[None, :], bones] = 1
    return dr.RestView(mask, np.zeros((H, W, 3)), np.tile([0, 0, 1.0], (H, W, 1)), w,
                       np.zeros((H, W)), np.zeros(0, np.int64), np.zeros(0, bool),
                       np.asarray(colour, np.float64))


def test_neutral_tint_takes_the_cast_off_steel_but_keeps_the_red():
    H, W = 12, 12
    mask = np.ones((H, W), bool)
    model = np.full((H, W, 3), 140.0)                  # grey steel...
    model[:, 8:] = (200, 30, 30)                       # ...and a red cape
    drawn = np.full((H, W, 3), (150, 135, 170), np.uint8)   # steel drawn lavender
    drawn[::2, :8] = (100, 85, 125)                    # with darker lavender shading
    drawn[:, 8:] = (190, 40, 50)
    out, (da, db) = dr.neutral_tint(drawn, mask, _view(mask, model, np.zeros((H, W), int)))
    from pixel_pipeline.color import rgb_to_lab
    before = rgb_to_lab(drawn[:, :8].astype(np.float32))
    after = rgb_to_lab(out[:, :8].astype(np.float32))
    assert np.hypot(da, db) > 8 and db < 0              # it was blue-purple
    cast = np.hypot(before[..., 1], before[..., 2])
    assert (np.hypot(after[..., 1], after[..., 2]) < 0.35 * cast).all()   # steel is grey again
    assert np.allclose(after[..., 0], before[..., 0], atol=1.5)   # same lightness, same shading
    assert np.array_equal(out[:, 8:], drawn[:, 8:])    # the saturated cape untouched


def test_a_blade_drawn_off_the_models_sword_still_follows_the_hand():
    H, W = 24, 16
    mask = np.zeros((H, W), bool)
    bones = np.zeros((H, W), int)
    colour = np.zeros((H, W, 3))
    mask[2:22, 3:9] = True                             # body + legs (bone 0 / leg bone 2)
    colour[2:22, 3:9] = (90, 90, 100)
    bones[12:22, 3:9] = 2
    mask[4:7, 9:12] = True                             # the hand (bone 1) right of the body
    bones[4:7, 9:12] = 1
    colour[4:7, 9:12] = (200, 160, 120)
    mask[7:14, 10] = True                              # the model's sword below it
    bones[7:14, 10] = 1
    colour[7:14, 10] = (220, 220, 230)
    view = _view(mask, colour, bones)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[mask] = colour[mask].astype(np.uint8)
    drawing = mask.copy()
    drawing[7:14, 10] = False                          # SDXL drew the blade 2 px to the left,
    drawing[7:20, 8] = True                            # longer, down over the leg
    rgb[7:20, 8] = (225, 225, 235)
    rgb[7:20, 7] = (20, 16, 28)                        # its outline against the leg
    ny, nx = dr.source_pixels(view, None, drawing)
    assert (bones[ny[7:20, 8], nx[7:20, 8]] != 1).sum() > 5      # plain: leg/body weights
    ny, nx = dr.source_pixels(view, rgb, drawing, held=[1])
    assert (bones[ny[7:20, 8], nx[7:20, 8]] == 1).all()          # now: the hand's
    leg = drawing.copy()
    leg[:, 7:] = False
    assert (bones[ny[leg], nx[leg]] == bones[leg]).all()          # the rest keeps its own


def test_mirror_and_twins(robot):
    rig, body, rest, poses, cams = robot
    assert [dr.twin_index(cams, k) for k in range(4)] == [2, None, 0, None]   # E <-> W
    m = np.zeros((4, 10), bool)
    m[:, 1] = True
    assert dr.mirror_frame(m, 5.0)[:, 8].all()         # x = 1 -> 2*5 - 1 - 1
    views = [dr.rest_view(body, c, rest) for c in cams]
    assert not dr.symmetric_pair(views, cams, 2, 0)    # the robot's one arm: not symmetric


def test_facing_choice_picks_the_drawing_that_faces_the_camera(robot):
    rig, body, rest, poses, cams = robot
    node = rig.names.index("root")
    turned = rest.worlds.copy()
    a = np.radians(90)                                 # the body spun a quarter turn
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    turned[node] = turned[node].copy()
    turned[node][:3, :3] = R @ turned[node][:3, :3]
    pose = dr.Pose(rest.mats, rest.shift, turned)
    j, face = dr.facing_choice(rest, rest, cams, 1, node)
    assert j == 1 and face == pytest.approx(1)
    j, face = dr.facing_choice(rest, pose, cams, 1, node)
    assert j != 1 and face == pytest.approx(1, abs=1e-6)
    assert dr.turned_away(rest, pose, cams[1], node) > 45


def test_recolour_palette_and_absent_words(robot):
    rig, body, rest, poses, cams = robot
    pal = np.array([[200, 30, 30], [30, 30, 200], [10, 10, 10]], np.uint8)
    out = dr.recolour_palette(pal, [["c81e1e", "1ec81e"]])
    assert out[0].tolist() == [30, 200, 30] and (out[1:] == pal[1:]).all()
    assert not dr.has_shield(body, rest)
    assert dr.auto_absent("a robot", "person", body, rest) == dr.SHIELD_WORDS
    assert "saddle" in dr.auto_absent("a grey wolf", "animal", body, rest)
    assert "collar" not in dr.auto_absent("a dog with a red collar", "animal", body, rest)


def test_cancel_resume_and_redraw_one_direction(comfy, tmp_path, monkeypatch):  # noqa: F811
    from contextlib import contextmanager

    from PIL import Image
    from pixel_pipeline import library
    fake, client = comfy
    monkeypatch.setattr(dr, "_gpu_temperature", lambda: None)
    fake.history_delay = 0
    monkeypatch.setattr(library, "style", lambda: Style(directions=4))
    rigged = write_robot(tmp_path / "robot.glb")
    keys = tmp_path / "keys"
    calls, held = [], []

    @contextmanager
    def gpu(what):
        held.append(what)
        yield
    with pytest.raises(dr.Cancelled):
        dr.drawn_sheets(rigged, "a robot", keys_folder=keys, frames=2, client=client,
                        cancel=lambda: len(fake.prompts) >= 3, gpu=gpu, progress=calls.append)
    assert len(fake.prompts) == 3 and len(held) == 3   # every drawing inside the GPU guard
    assert json.loads((keys / "keys.json").read_text())["partial"]   # same scale on resume
    assert any("min left" in c for c in calls)         # an ETA once one drawing is timed
    dr.drawn_sheets(rigged, "a robot", keys_folder=keys, frames=2, client=client,
                    out_png=tmp_path / "sheet.png")
    full = 3 * len(dr.CANDIDATES) + len(dr.BACK_CANDIDATES)
    assert len(fake.prompts) == full                   # carried on, nothing drawn twice
    seeds = json.loads((keys / "keys.json").read_text())["seeds"]
    before = np.asarray(Image.open(keys / "key_S.png")).copy()
    dr.drawn_sheets(rigged, "a robot", keys_folder=keys, frames=2, client=client,
                    out_png=tmp_path / "sheet.png", redraw=["E"])
    assert len(fake.prompts) == full + len(dr.CANDIDATES)   # only E again
    again = json.loads((keys / "keys.json").read_text())["seeds"]
    assert again["E"] != seeds["E"] and again["S"] == seeds["S"]
    assert (keys / "old" / "drawn_E_o60.png").exists()
    assert np.array_equal(np.asarray(Image.open(keys / "key_S.png")), before)


def test_library_default_drawn_reskin_and_restyle(comfy, tmp_path, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from PIL import Image
    from pixel_pipeline import assets as A
    from pixel_pipeline import autorig, generate, library, reskin
    fake, client = comfy
    monkeypatch.setattr(generate, "ComfyClient", lambda *a, **k: client)
    monkeypatch.setattr(dr, "_gpu_temperature", lambda: None)
    fake.history_delay = 0
    monkeypatch.setattr(library, "style", lambda: Style(directions=4))
    monkeypatch.setattr(library, "add_project_clips", lambda *a, **k: [])
    monkeypatch.setattr(autorig, "auto_rig", lambda model, out: write_robot(out))
    model = write_robot(tmp_path / "robot_model.glb")
    monkeypatch.setenv("PIXEL_DRAWN", "0")
    plain = library.add_character("Plain", model, prompt="a robot")
    assert not library.is_drawn(plain) and not fake.prompts       # no ComfyUI: the 3D render
    monkeypatch.setenv("PIXEL_DRAWN", "1")
    ch = library.add_character("Bot", model, prompt="a robot")    # default: drawn
    assert library.is_drawn(ch) and fake.prompts
    assert (ch.folder / "keys" / "model.sha1").exists()
    assert "draw it again" in A._restyle_how(SimpleNamespace(kind="character", folder=ch.folder,
                                                             extra={}))
    n = len(fake.prompts)
    from pixel_pipeline.reskin import colour_hex, image_colours
    cols, counts = image_colours(np.asarray(Image.open(ch.sheet).convert("RGBA")))
    top = colour_hex(cols[counts.argmax()])
    skin = reskin.reskin(ch, "Bot Blue", palette_map={top: "3050c0"})
    assert skin.prompt == "a robot"
    assert json.loads((skin.folder / "keys" / "keys.json").read_text())["recolour"]
    sheet = np.asarray(Image.open(skin.sheet).convert("RGBA")).copy()
    library.redraw(skin, ["W"])                        # bent again: the re-skin's colours kept
    after = np.asarray(Image.open(skin.sheet).convert("RGBA"))
    old_cols = {tuple(c) for c in image_colours(sheet)[0]}
    new_cols = {tuple(c) for c in image_colours(after)[0]}
    assert len(new_cols & old_cols) >= 0.8 * len(new_cols)
    assert len(fake.prompts) == n + len(dr.CANDIDATES)


def test_model_studio_drawn_default_stop_and_redraw(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline import library
    from pixel_pipeline.ui.model_studio import ModelStudio
    monkeypatch.setenv("PIXEL_DRAWN", "1")
    win = ModelStudio()
    win.sync = True
    assert win.drawn.isChecked()                       # ComfyUI here: drawn by default
    monkeypatch.setenv("PIXEL_DRAWN", "0")
    assert not ModelStudio().drawn.isChecked()
    ch = library.Character("Bot", folder=tmp_path / "bot")
    (ch.folder / "keys").mkdir(parents=True)
    (ch.folder / "keys" / "keys.json").write_text(json.dumps({"camera": {"directions": 8}}))
    seen = {}

    def fake_redraw(c, directions, progress=None, cancel=None, gpu=None):
        seen.update(ch=c, directions=directions, cancelled_before=cancel())
        win.stop()                                     # the Stop button while it draws
        seen["cancelled_after"] = cancel()
        return c
    monkeypatch.setattr(library, "redraw", fake_redraw)
    monkeypatch.setattr(win, "open_character", lambda c: seen.update(opened=c))
    win.redraw_direction(ch, ["NE"])
    assert seen["directions"] == ["NE"] and seen["opened"] is ch
    assert not seen["cancelled_before"] and seen["cancelled_after"]
    assert not win.stop_btn.isVisible() and win._cancel is None
    from pixel_pipeline.ui.model_studio import _drawn_directions
    assert _drawn_directions(ch) == 8
    win.close()


def test_animals_are_drawn_from_the_side():
    from pixel_pipeline.turnaround import direction_names
    names = direction_names(8)
    cams = [dr.Camera(k * 45.0, 30.0, 1.0, 0.0, 0.0, 1, 1) for k in range(8)]
    assert names[dr.draw_order(names, cams, "E")[0]] == "E"
    assert [names[k] for k in dr.draw_order(names, cams, "S")[:4]] == ["S", "E", "W", "N"]
    assert dr.view_refs("S", 90.0, {"E", "W"}, "E") == ["E"]
    assert dr.view_refs("N", 270.0, {"E", "W"}, "E") == ["E", "W"]
    assert dr.view_refs("NW", 225.0, {"E", "W", "N"}, "S") == ["N", "W"]
    assert dr.REFERENCES["animal"] == "E"
